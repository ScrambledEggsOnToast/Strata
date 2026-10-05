// include/strata/prefill/prefill.hpp - plan v0.3 P5: batched prompt processing.
//
// The prompt's positions [pos0, pos0 + n) are processed in chunks of `chunk` tokens through all 48 layers, leaving
// the session state (GDN recurrence and conv state, QSA KV pools and indexer, PLE history) where the token path
// would have left it; the decode loop then continues with the next token.  Per layer: the projections are
// tensor-core GEMMs (quantized weights dequantized to FP16 on the fly, BF16 weights as they are), the recurrences
// walk the chunk inside one kernel, and the routed experts are grouped by expert: resident ones are read from the
// VRAM tier, the others streamed from the host arena through a pinned ring on a copy stream.
//
// Requires the native weights (`--native`): every quantized projection must carry its GGUF blocks.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>

namespace strata::core { class PeerExperts; }

namespace strata::prefill {

struct PrefillStats {
    int64_t tokens = 0;
    int64_t chunks = 0;
    double ms_total = 0;
    double ms_experts_host = 0;     ///< host time staging non-resident experts
    int64_t experts_streamed = 0;   ///< expert blobs copied host -> device
    int64_t experts_dma = 0;        ///< ...of which straight from the pinned arena (no CPU copy)
    int64_t experts_resident = 0;   ///< expert-layer groups served from the VRAM tier
    double ms_ple = 0;
};

}  // namespace strata::prefill
namespace strata::core { class MtpDrafter; }
namespace strata::prefill {

/// Inputs available before weights/session allocation. n_pages is the primary QSA state's logical page count
/// (ceil(max_cells / 4)), NOT its resident slot count. Snapshot environment choices with allocation_config().
struct AllocationConfig {
    int64_t max_cells = 0, n_pages = 0, chunk = 0;
    int64_t layer_begin = 0, layer_end = -1; ///< Stage validation; device scratch remains shared/all-path sized.
    int kv_mode = 0;
    bool kv_int8 = false, kv_q4 = false, kv_stage_own = false, gr_unfused = false, mmq = true;
    int ring = 8; ///< Snapshot ring_slots(chunk): 8 routed slots, otherwise clamped 16..512 streaming slots.
    int stager_ring = 16; ///< Snapshot of the host copy ring for unpinned blobs (STRATA_STAGER_RING, clamped
                          ///< 2..256 exactly as the stager reads it at init).
    int device_cc = 0, device_sms = 0;
    uint64_t device_shared_bytes = 0;
};
struct AllocationBytes {
    uint64_t loanable_device = 0;  ///< Exact aligned bump region, including GEMM and bounded MMQ workspace.
    uint64_t owned_device = 0;     ///< Token ids, staging identity table, and optionally separately owned KV stage.
    // Every take(payload) requests ((payload + 511) & ~255): ceil(payload / 256)*256 + 256.
    // All three attention/MoE overlays are counted by their actual take sequence; only their maximum is allocated.
    // A profiled/undersized/empty expert cache changes none of these totals. A partial loan cannot price the full
    // region as borrowed: init requires a 256-byte aligned loan holding at least loanable_device bytes.
    uint64_t standalone_device = 0;///< loanable_device + owned_device; valid even when no cache slots can be lent.
    uint64_t mmq_workspace = 0;    ///< Included in loanable_device; configuration-derived maximum, not a heuristic.
    /// Fixed explicit host payload, excluding separately priced stager blob slots, PLE staging pairs,
    /// routing table heads, step logs and split hand-offs. Opaque driver storage and thread stacks remain
    /// a separate unknown class; the counted dynamic host storage is host_dynamic below.
    /// Counts token staging chunk*4; routing counts (2*n_expert+1)*4; stager_ring pinned flags;
    /// the mapped bounds tail (n_expert+1+ceil(n_expert/16)*17)*4 even without MMQ, because init
    /// reserves it regardless; and the same-sized bounds staging payload when any layer runs MMQ.
    /// bounds_host is initially sized to its maximum so per-layer resizing cannot grow capacity.
    uint64_t host_payload = 0;
    /// The dynamic host payload's counted fixed storage, charged at the geometry bound each buffer is
    /// allocated to, once, at init (from stager_max_jobs and the config), and never grown:
    /// - the stager's job list (the stage's layers * n_expert entries) and its ready flags (twice that),
    ///   refilled in place per chunk or layer and published only after finish() released the previous
    ///   list, so a late worker can never read an overwritten job;
    /// - the stager ring's slot-pointer and DMA-event-handle arrays and its fallback slot containers
    ///   (stager_ring entries each);
    /// - the whole-chunk stream plan (one entry per streamed expert-layer pair) and its layer offsets,
    ///   resident in both modes, so a relayout's routed/stream transition stays inside this bound;
    /// - the MoE grouping arrays (order/fill and the ring-slot/job indices, n_expert entries each),
    ///   refilled in place per layer;
    /// - the draft layer's token staging (the init chunk's tokens; exact per call, never beyond this);
    /// - the KV-streaming init identity page table (n_pages entries, only when kv_mode == 1).
    /// Every count is enforced - the fillers refuse by name at the bound - and every buffer holds exactly
    /// count * sizeof(T) for trivially destructible element types; sizes come from sizeof, never hard-coded
    /// ABI numbers.  NOT included, because nothing documents their sizes (the caller's unknown class):
    /// host allocator block headers or any other heap/driver metadata around these owned arrays,
    /// cudaEvent_t driver objects, resident thread stacks, and the container-managed init-lifetime
    /// device-pointer list, thread-handle container and pinned-flag vector.
    uint64_t host_dynamic = 0;
};

class Prefill {
public:
    /// E-9: the draft layer's K/V for prompt cells [cell0, cell0 + n) from their final residual rows `R_rows`
    /// (device) and `next_tokens` (host: the token at cell+1), in batches through this path's GEMMs and its idle
    /// scratch - call it from on_chunk.  false with `err` empty: not applicable here (a ring or hybrid K/V, another
    /// device, too little scratch; STRATA_MTP_BATCH=0), the caller runs the drafter's own pass.  Not bit-identical to
    /// that pass (FP16 GEMMs instead of Q8_1 activations): the drafts may differ, never the target's tokens' logits.
    bool draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                  std::string& err);
    Prefill();
    ~Prefill();
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    /// Frees every buffer, stream and event `init` made (as the destructor does) and starts over empty, so `init` can
    /// run again - with a smaller chunk when the first one did not fit.  The stage range (`set_stage`) and the
    /// callbacks stay.  The device `init` ran on must be current.
    void reset();

    /// `host_res`: the static residency table (n_layers x n_expert, slot or -1) or null; `cache` its slots.
    /// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
    /// lent for the prompt and refilled after it); null = allocate normally.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0);

    /// With borrowed buffers: lay them out again for chunks of `chunk` tokens (at most `init`'s) in `borrow` - a
    /// request lends only the slots its prompt needs.  The stream must be idle (between prompts).
    bool relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err);
    int64_t chunk() const;

    /// The share of the streamed experts' bytes DMA-able straight from pinned RAM (1 = all).  Sizes the streamed
    /// ring (a big one only pays when the copy engine, not the host copies, is the limit); set before bytes_needed.
    static void set_pinned_share(double share);
    static double pinned_share();
    /// #340: the streamed ring's slot count for chunks that stream every expert, instead of the pinned-share rule
    /// (0 = that rule). Set before any `bytes_needed`/`init` (both count the ring); STRATA_PREFILL_RING still wins.
    static void set_ring_override(int slots);
    /// #583: the auto chunk scan's byte-budget ring, for chunks above `small_max` (0.1.39's auto chunk: a prompt that
    /// fits it keeps 0.1.39's ring).  0 slots = none.  A layer split's set_ring_override and STRATA_PREFILL_RING win.
    static void set_ring_budget(int slots, int64_t small_max);

    /// Snapshot the same runtime switches used by init. Queries device properties, but allocates no device memory.
    static bool allocation_config(int64_t max_cells, int64_t chunk, int kv_mode, bool kv_int8, bool kv_q4,
                                  AllocationConfig& out, std::string& err);
    /// The dynamic stager's fixed job bound for a stage: the stage's layer range times n_expert - the most
    /// expert-layer pairs a chunk's plan or job list can hold - checked against the claim word's 16-bit job
    /// field (refuse, never clamp; this also rules out any product overflow).  init sizes the counted stager
    /// and stream-plan storage from this, and allocation_needed prices host_dynamic from the same value.
    /// false (with max_jobs = 0): the stage range or the expert count is outside the encodable range.
    static bool stager_max_jobs(const core::ModelGeometry& g, const AllocationConfig& c, int64_t& max_jobs);
    /// Pure checked arithmetic; no SessionState, CUDA allocation, global artifact layout, or cache dependency.
    /// No loan is deducted: callers may subtract loanable_device ONLY after guaranteeing that region is lent.
    /// Includes explicit CUDA allocations, not opaque driver/cuBLAS handle internals (price those separately).
    static bool allocation_needed(const core::ModelGeometry& g, const AllocationConfig& config,
                                  const kernels::cpu::ExpertLayout& experts, AllocationBytes& out, std::string& err);
    /// Existing initialized-session consumers need the bump region only. Invalid/overflow returns UINT64_MAX.
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk);

    /// The same without the streamed ring: what the chunk's own buffers cost.  The auto chunk scan sizes the chunk
    /// first and hands the ring what the chunk leaves over, so it needs the chunk priced on its own.
    static uint64_t bytes_needed_no_ring(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk);

    /// The streamed ring's byte budget as a slot count for this pack (the measured slot count x Q2_0's blob, over
    /// max_blob, never past ring_cap()):
    /// what the auto chunk scan treats as a full ring.  A slot is one whole blob, so a pack with bigger blobs than
    /// Q2_0's gets fewer of them for the same bytes - 384 on Q2_0, 199 on a 2.54 MiB-blob IQ3_S pack.
    static int64_t ring_max_slots();
    /// 0.1.39's ring for this PC (1024 fused / 384 pinned / 96), within ring_cap().
    static int64_t ring_default_slots();
    /// #583: the ring the auto scan keeps full, given the chunk 0.1.39's rule picked: its byte budget where that rule's
    /// chunk was small (< 6144), else 0.1.39's ring (measured: shrinking it for a bigger chunk lost there).
    static int64_t ring_cap_for(int64_t old_chunk);

    /// What the ring actually resolves to for a chunk of `chunk` tokens, after the override, STRATA_PREFILL_RING
    /// and the pinned-share rule - the slot count `init` lays out.  The engine reports it on its INFO line so the
    /// Monitor tab shows the pair the run really got, not what it asked for.
    static int64_t ring_slots_for(int64_t chunk);

    /// 0.1.39b (#583, the default): the ring as a byte budget, the loan's corrected count and the auto chunk scan that
    /// keeps the ring full.  STRATA_RING_BYTES=0: 0.1.39's ring, loan and chunk list.
    static bool ring_bytes_enabled();

    /// Positions [pos0, pos0 + n) holding `tokens`; `ss.ple_prev` must be the two tokens before pos0 (oldest
    /// first, -1 for none) and is advanced to the last two of these.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    const PrefillStats& stats() const { return stats_; }

    /// multi-GPU: the experts the peer GPU holds are computed THERE for every prompt chunk (up to `cap_rows` routed
    /// rows per layer; the rest of the peer's experts are read by this GPU over P2P).  Allocates the peer's buffers for
    /// chunks of up to init's `chunk` tokens.  Needs P2P between the two cards.
    bool set_peer(core::PeerExperts* peer, int64_t cap_rows, std::string& err);

    /// Plan v0.3 P6: called after every chunk with the chunk's final multi-stream residual rows (device,
    /// T x hc*n_embd, valid until the next chunk) and the chunk's first position; the MTP draft layer builds its
    /// K/V from them.  The prefill stream is synchronized before the call.
    std::function<bool(const float* R_rows, int64_t T, int64_t pos0, std::string& err)> on_chunk;

    /// Layer split: called by every stage when it has read a chunk, with the position reached, while its own state
    /// is still at that chunk's end (its stream synchronized; the last stage calls it just before `on_chunk`).  An
    /// earlier stage is a chunk or more ahead of the last one by the time `on_chunk` runs, so this is where a
    /// mid-prompt checkpoint takes each stage's part.  Runs on that stage's thread, with its device current.
    std::function<bool(int64_t done, std::string& err)> on_stage_chunk;

    /// Checked before every chunk: true stops the prompt early (`run` returns false with err "cancelled").
    std::function<bool()> should_stop;

    /// The vision path: HOST rows (n_embd floats) indexed by absolute position, read in place of the token
    /// embedding where non-null (an image's <|image_pad|> cells).  Null (default): every position embeds its token.
    const float* const* embd_rows = nullptr;

    /// LAYER SPLIT (multi-GPU): this prompt path runs layers [layer_begin, layer_end) (-1: to the last) on the
    /// device `init` runs on.  A stage that does not start at layer 0 reads each chunk's residual rows from the
    /// previous stage instead of embedding the tokens; a stage that does not end at the last layer copies its rows
    /// to pinned host buffers (two, allocated by `init`) and runs `next` on them - on a thread, so the next stage
    /// reads chunk c while this one reads chunk c + 1.  `on_chunk` belongs on the last stage.  Set before `init`.
    void set_stage(int64_t layer_begin, int64_t layer_end, Prefill* next) {
        stage_lb_ = layer_begin; stage_le_ = layer_end; next_ = next;
    }

    /// LAYER SPLIT: `helper` is another stage's prompt path (another GPU).  A prompt of one chunk runs the stages one
    /// after the other, so while this stage reads it the helper's GPU idles: it then streams a share of this stage's
    /// non-resident experts over its own PCIe link into its own (lent) prompt buffers, computes their rows, and sends
    /// them back - the --peer-device peer's streaming, without P2P (activations and rows through mapped host memory,
    /// read by copy kernels).  Only on the MMQ prompt path, only for chunks of stream_all_min() tokens and more, up to
    /// the size where the measured share stops paying.  Opt-in: STRATA_PREFILL_HELP=1 (not bit-identical to the default).  Both stages must have
    /// run `init`.
    bool set_stage_helper(Prefill* helper, std::string& err);

private:
    // Stage-1 pipeline: intermediate stages return after handing their chunk to
    // the direct successor. The public run() drains the chain once at prompt end.
    bool run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);
    bool drain_pipeline(std::string& err);

    int64_t stage_lb_ = 0, stage_le_ = -1;
    Prefill* next_ = nullptr;
    Prefill* helper_ = nullptr;         ///< set_stage_helper
    bool single_chunk_ = false;         ///< a later stage: the prompt is one chunk (set by the stage before)
    bool bind_stage_helper(int64_t T);  // binds the helper's buffers for a one-chunk prompt of T tokens
    const float* hand_in_ = nullptr;    ///< the previous stage's rows of the chunk being read (host, pinned)

    std::string next_err_;
    std::future<bool> next_run_;
    int hand_buf_ = 0;

    bool carve(std::size_t T, void* alloc);   // the device buffers of a chunk (prefill.cpp's Alloc)
    void release();                          // the destructor's cleanup (also `reset`'s)
    struct Impl;
    static bool carve_layout(Impl& m, std::size_t T, void* alloc, void* owned_alloc);
    static bool carve_fixed(Impl& m, void* alloc);
    std::unique_ptr<Impl> impl_;
    PrefillStats stats_;
};

}  // namespace strata::prefill
