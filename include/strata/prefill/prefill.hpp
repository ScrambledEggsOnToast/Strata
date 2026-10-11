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

#include <cstdint>
#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <string>

namespace strata::core { class PeerExperts; }
namespace strata::kernels::cpu { class ExpertPool; }

namespace strata::prefill {

struct PrefillStats {
    int64_t tokens = 0;
    int64_t chunks = 0;
    double ms_total = 0;
    double ms_experts_host = 0;     ///< host time staging non-resident experts
    int64_t experts_streamed = 0;   ///< expert blobs copied host -> device
    int64_t experts_dma = 0;        ///< ...of which straight from the pinned arena (no CPU copy)
    int64_t experts_resident = 0;   ///< expert-layer groups served from the VRAM tier
    int64_t experts_cpu = 0;        ///< ...computed on the CPU pool instead of streamed (STRATA_PREFILL_CPU_SHARE)
    double cpu_share = 0;           ///< ...the share of the streamed ones it took last (measured by default)
    double ms_ple = 0;
};

}  // namespace strata::prefill
namespace strata::core { class MtpDrafter; }
namespace strata::prefill {

// ============================ FORK: pre-allocation admission (HET-017) ============================
// Everything below this marker is the fork's, not upstream's.  It exists because the admission gate prices a
// prompt configuration BEFORE a SessionState exists, while upstream's own counter takes the session.  The two must
// be one implementation: `bytes_needed` (session) and `bytes_needed_from` (facts) share the same body, so a change
// to the allocation sequence cannot leave the price behind.

/// The facts the prompt path's allocation depends on, read from the session or supplied by a caller that has none.
struct PrefillFacts {
    int64_t max_cells = 0;   ///< the primary QSA state's cell budget
    int64_t n_pages = 0;     ///< ...and its resident page count (KV streaming's identity table)
    int kv_mode = 0;         ///< 0: device KV, 1: streaming/native, 2: hybrid
    bool kv_int8 = false, kv_q4 = false, kv_hybrid = false;
    /// The session's own answer, for `bytes_needed`'s sake.  A session with no QSA states gives a zeroed set.
    static PrefillFacts of(const core::SessionState& ss);
};

/// Inputs available before weights/session allocation. n_pages is the primary QSA state's logical page count
/// (ceil(max_cells / 4)), NOT its resident slot count. Snapshot environment choices with allocation_config().
struct AllocationConfig {
    int64_t max_cells = 0, n_pages = 0, chunk = 0;
    int64_t layer_begin = 0, layer_end = -1; ///< Stage validation; device scratch remains shared/all-path sized.
    int kv_mode = 0;
    bool kv_int8 = false, kv_q4 = false, kv_hybrid = false, kv_stage_own = false, gr_unfused = false, mmq = true;
    int ring = 8; ///< Snapshot of ring_slots_for(chunk) - the ring `init` lays out, clamped 16..ring_cap().
    int stager_ring = 16; ///< Snapshot of the host copy ring for unpinned blobs (STRATA_STAGER_RING, clamped
                          ///< 2..256 exactly as the stager reads it at init).
    int device_cc = 0, device_sms = 0;
    uint64_t device_shared_bytes = 0;
    PrefillFacts facts() const {
        PrefillFacts f;
        f.max_cells = max_cells; f.n_pages = n_pages; f.kv_mode = kv_mode; f.kv_int8 = kv_int8; f.kv_q4 = kv_q4;
        f.kv_hybrid = kv_hybrid;
        return f;
    }
};
struct AllocationBytes {
    uint64_t loanable_device = 0;  ///< Exact aligned bump region, including the streamed ring and MMQ workspace.
    uint64_t owned_device = 0;     ///< Token ids, the streaming identity table, and optionally an owned KV stage.
    /// Every take(payload) requests ((payload + 511) & ~255): ceil(payload / 256)*256 + 256.
    /// A partial loan cannot price the full region as borrowed: init requires a 256-byte aligned loan holding at
    /// least loanable_device bytes.
    uint64_t standalone_device = 0;///< Owned allocation sequence, each allocation rounded to a 2 MiB granule.
    uint64_t mmq_workspace = 0;    ///< Included in loanable_device; configuration-derived maximum, not a heuristic.
    /// Fixed explicit HOST payload (token staging, routing tables, the mapped bounds tail, stager ring flags).
    /// Opaque driver storage and thread stacks remain a separate unknown class; host_dynamic below is the
    /// counted storage the stager allocates for its jobs.
    uint64_t host_payload = 0;
    /// Host storage whose size the engine's own bound fixes (the stager's job list and its ready flags, the MoE
    /// grouping arrays, the stream plan, the per-layer MMQ table, the draft layer's token staging, the streaming
    /// identity page table), priced from the same bound `stager_max_jobs` gives `init`.  NOT included, because
    /// nothing here bounds them: allocator block headers and other heap/driver metadata around these arrays,
    /// cudaEvent_t driver objects, resident thread stacks, and the container-managed init-lifetime tables.
    uint64_t host_dynamic = 0;
};

// ========================== END FORK BLOCK (admission structs) ==========================

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
    bool bind_request(uint64_t request, std::string& err);
    bool context_idle(std::string& err) const;

    /// The share of the streamed experts' bytes DMA-able straight from pinned RAM (1 = all).  Sizes the streamed
    /// ring (a big one only pays when the copy engine, not the host copies, is the limit); set before bytes_needed.
    static void set_pinned_share(double share);
    static double pinned_share();
    /// The chunk size from which a chunk streams every expert the GPU does not hold (1024; STRATA_PREFILL_STREAM_MIN).
    static int64_t stream_all_min_tokens();
    /// #340: the streamed ring's slot count for chunks that stream every expert, instead of the pinned-share rule
    /// (0 = that rule). Set before any `bytes_needed`/`init` (both count the ring); STRATA_PREFILL_RING still wins.
    static void set_ring_override(int slots);
    /// #583: the auto chunk scan's byte-budget ring, for chunks above `small_max` (0.1.39's auto chunk: a prompt that
    /// fits it keeps 0.1.39's ring).  0 slots = none.  A layer split's set_ring_override and STRATA_PREFILL_RING win.
    static void set_ring_budget(int slots, int64_t small_max);

    /// Device bytes `init` needs for a chunk of `chunk` tokens (what a borrowed region must hold).
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk);

    /// The same without the streamed ring: what the chunk's own buffers cost.  The auto chunk scan sizes the chunk
    /// first and hands the ring what the chunk leaves over, so it needs the chunk priced on its own.
    /// What `init` allocates when the prompt path OWNS its buffers: each cudaMalloc rounded up to a 2 MiB page and the
    /// ring as one allocation (the startup sizing of a cache without a loan).
    static uint64_t bytes_needed_owned(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk);
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

    // ---- FORK (HET-017): the admission gate's entry points, priced from the facts above ----
    /// Snapshot the same runtime switches init uses. Queries device properties, allocates no device memory.
    /// `device_ordinal` < 0 reads the CURRENT device; otherwise that visible device's own MMQ hardware
    /// inputs (cc/SMs/shared memory) are snapshotted, so a layer split prices each stage's own workspace.
    static bool allocation_config(int64_t max_cells, int64_t chunk, int kv_mode, bool kv_int8, bool kv_q4,
                                  AllocationConfig& out, std::string& err, bool kv_hybrid = false,
                                  int device_ordinal = -1);
    /// The dynamic stager's fixed job bound for a stage: the stage's layer range times n_expert - the most
    /// expert-layer pairs a chunk's plan can hold - checked against the claim word's 16-bit job field (refuse,
    /// never clamp; this also rules out any product overflow).
    static bool stager_max_jobs(const core::ModelGeometry& g, const AllocationConfig& c, int64_t& max_jobs);
    /// Pure checked arithmetic; no SessionState, CUDA allocation, global artifact layout, or cache dependency.
    /// No loan is deducted: callers may subtract loanable_device ONLY after guaranteeing that region is lent.
    /// Includes explicit CUDA allocations, not opaque driver/cuBLAS handle internals (price those separately).
    static bool allocation_needed(const core::ModelGeometry& g, const AllocationConfig& config,
                                  const kernels::cpu::ExpertLayout& experts, AllocationBytes& out, std::string& err);
    /// FORK (HET-022): the exact device and pinned-host bytes ONE resident prompt helper needs for a chunk of
    /// `chunk` tokens, from the same term sequence `set_prompt_helpers` allocates (the allocator checks itself
    /// against this price, so a change to one cannot leave the other behind).  `p2p` selects the route the
    /// declaration will get and `helper_device` the owner's own MMQ workspace (the same `mmq::device_config` /
    /// `mmq::workspace_bytes` pair the allocator binds); `cap_rows` is the rows route's cap (the `!p2p` bound is
    /// the chunk's own routing).  Charges no allocation and reads no SessionState, so admission can price the
    /// helper before anything is created. Includes device and mapped-host bounds for one compact group per
    /// expert; pageable grouping vectors stay within the caller's named per-owner runtime allowance.
    /// The MMQ geometry is the current device's (the plan prices stages with their own device current).
    static bool helper_allocation(const core::ModelGeometry& g, int64_t chunk, int64_t cap_rows, bool p2p,
                                  int helper_device, bool resident_only, uint64_t& device_bytes,
                                  uint64_t& host_bytes, std::string& err);

    /// `bytes_needed`'s body, callable without a session: one implementation, two callers.  `ring_slots_override`
    /// >= 0 prices that many ring slots instead of the engine's own choice, so admission can price the ring a
    /// configuration names (`allocation_config` snapshots it from ring_slots_for).
    static uint64_t bytes_needed_from(const core::ModelGeometry& g, const PrefillFacts& f, int64_t chunk,
                                      int ring_slots_override = -1, uint64_t ring_blob = 0,
                                      int stage_own_override = -1, bool owned_pages = false);
    /// FORK: the environment switches the price snapshot must agree with `init`/`carve` on - one rule, not two
    /// copies of it.  `ring_hard_cap` is the array bound the ring can never exceed (RING_MAX / ring_cap()).
    static bool kv_stage_own_env();
    static bool gr_unfused_env();
    static int stager_ring_env();
    static int ring_hard_cap();
    /// FORK: the bytes the owned KV stage adds when STRATA_KV_STAGE_OWN is set (the loan counter skips them, as
    /// they are not lent).  0 when the stage is not owned or the KV mode does not stream.
    static uint64_t owned_stage_bytes(const core::ModelGeometry& g, const PrefillFacts& f);
    /// FORK: the host-side price, next to the allocations it prices: `payload` is the fixed explicit staging
    /// (token staging, routing heads, the stager's pinned flags, the mapped bounds tail) and `dynamic` the
    /// counted storage whose size the engine's own bound fixes (the stager's job list and ready flags, the MoE
    /// grouping arrays, the per-layer MMQ table, the draft layer's token staging, the streaming page table).
    static bool host_allocation_from(const core::ModelGeometry& g, const PrefillFacts& f, int64_t chunk,
                                     int stager_ring, bool mmq, int64_t max_jobs, uint64_t& payload,
                                     uint64_t& dynamic);
    // ---- END FORK (admission) ----

    /// Positions [pos0, pos0 + n) holding `tokens`; `ss.ple_prev` must be the two tokens before pos0 (oldest
    /// first, -1 for none) and is advanced to the last two of these.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    const PrefillStats& stats() const { return stats_; }

    /// The CUDA device this prompt path's buffers live on (-1 before `init`), for callers that price or route
    /// work per stage.
    int device() const;

    /// FORK (HET-022): one RESIDENT EXPERT OWNER's geometry, as the prompt path needs it - the CUDA device its
    /// cache lives on, that cache, and its (layer, expert) -> slot table.  A cache-backed descriptor, not a class:
    /// core::PeerExperts (--peer-device) and core::RemoteExperts (--expert-cache-device1..3) both render it, so one
    /// prompt path serves every resident owner.  `residency[l * n_expert + e]` is that cache's slot or -1.
    struct PromptHelper {
        int device = -1;
        const core::ExpertCache* cache = nullptr;
        const int32_t* residency = nullptr;
        // Dedicated resident ownership: FP32 weighted partials, no weight streaming.
        // False retains the existing peer tier's experimental transfer defaults.
        bool resident_only = false;
        // The owner must already have enabled access in both directions.
        bool p2p_enabled = false;
    };

    /// multi-GPU (HET-022): listed owners compute each chunk's routed rows for the experts their cache holds.
    /// Empty helpers means unassisted execution; a matched native helper-off control instead retains the owners
    /// with execute=false. This stage's authoritative ids/weights/slot/src tables decide every routed row.
    /// Resident owners apply those weights once and return FP32 partial sums for
    /// the primary's final combine. Their expert weights remain in their own VRAM.
    /// Dedicated remote owners use portable mapped host transport; a peer tier may
    /// use P2P only when its owner has already enabled access in both directions.
    /// One owner serves a layer at a time (the static residency decides which); several owners therefore serve
    /// different layers, and every helper GPU among them does prompt expert work.  Buffers are allocated for
    /// chunks of up to init's `chunk` tokens; call after `init` (the same allocator's sizes, not estimates).
    /// execute=false records ownership for the matched native-grouped control
    /// without allocating or launching helper work.
    /// The caches/residency tables must remain valid and static until reset or replacement.
    bool set_prompt_helpers(const std::vector<PromptHelper>& helpers, int64_t cap_rows, bool execute, std::string& err);
    /// Read only after loading the native layout: this freezes the prompt's quantization plan on first use.
    static bool native_grouped_enabled();

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

    /// The CPU expert pool (decode's, idle while a prompt is read). With STRATA_PREFILL_CPU_SHARE set, a chunk below
    /// stream_all_min() tokens - an agent's tool output - hands it the non-resident experts routed by at most MAXT of
    /// its tokens, fewest first, up to a share of the experts it would stream (`auto`: measured, where both sides end
    /// together). The CPU reads them from RAM (the arena, the page-locked copy, or the mapped experts.bin's pages)
    /// while the rest come over PCIe; their rows go to Dm's tail, as a peer's do. Not bit-identical to the GPU's rows
    /// (the CPU's own activation format). Unset (default) or null: every expert on the GPU. Only for a pool no other
    /// thread runs meanwhile (no batch slots); on a layer split every stage may have it - one stage at a time takes it
    /// for a chunk. Set before `init`.
    void set_cpu_pool(kernels::cpu::ExpertPool* pool);
    /// With STRATA_PREFILL_CPU_SHARE set and `applies` (a pool will be set): the chunks staged after their routing - the
    /// only ones the share applies to - go up to STRATA_PREFILL_CPU_SHARE_MAX tokens (default 3072) instead of 1024.
    /// Before any `bytes_needed` (the ring and the small-chunk buffers follow the limit); `init` also sets it.
    /// `by_default`: this path is one the share was measured on (CUDA, one GPU, no batch slots), so with the variable unset
    /// the share is on for chunks below 1024 tokens (STRATA_PREFILL_CPU_SHARE=0 turns it off).
    static void arm_cpu_share(bool applies, bool by_default = false);

private:
    static uint64_t bytes_needed_impl(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                      bool owned_pages);
    kernels::cpu::ExpertPool* cpu_pool_ = nullptr;   ///< set_cpu_pool
    // Stage-1 pipeline: intermediate stages return after handing their chunk to
    // the direct successor. The public run() drains the chain once at prompt end.
    bool run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);
    bool drain_pipeline(std::string& err);
    std::atomic<bool> public_running_{false};

    int64_t stage_lb_ = 0, stage_le_ = -1;
    Prefill* next_ = nullptr;
    Prefill* helper_ = nullptr;         ///< set_stage_helper
    bool single_chunk_ = false;         ///< a later stage: the prompt is one chunk (set by the stage before)
    bool bind_stage_helper(int64_t T);  // binds the helper's buffers for a one-chunk prompt of T tokens
    /// FORK (HET-022): allocate one resident owner's prompt helper (streams, events, compact MMQ buffers, mapped
    /// host crossings) on its device and append it to the engaged set; false with `err` set when it cannot serve.
    bool add_prompt_helper(const PromptHelper& owner, int64_t cap_rows, std::string& err);
    const float* hand_in_ = nullptr;    ///< the previous stage's rows of the chunk being read (host, pinned)

    std::string next_err_;
    std::future<bool> next_run_;
    int hand_buf_ = 0;

    bool carve(std::size_t T, void* alloc);   // the device buffers of a chunk (prefill.cpp's Alloc)
    void release();                          // the destructor's cleanup (also `reset`'s)
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PrefillStats stats_;
};

}  // namespace strata::prefill
