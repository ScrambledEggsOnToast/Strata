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

#include "strata/core/dflash2_capture.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/prefill/segments.hpp"
#include "strata/prefill/handoff.hpp"

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
    int64_t experts_direct = 0;     ///< ...of which MMQ read in their cache slots, not gathered (STRATA_MMQ_DIRECT=1)
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
    bool cpu_share = false; ///< Resolved pool/share eligibility before the ring/layout snapshot.
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
    /// CPU-share pinned input/output, both retained activation formats, exact job/selection/work arrays.
    /// Separate from host_payload/host_dynamic and opaque allocator/driver/thread metadata.
    uint64_t host_cpu_share = 0;
};

// ========================== END FORK BLOCK (admission structs) ==========================

class Prefill {
public:
    /// E-9: the draft layer's K/V for prompt cells [cell0, cell0 + n) from their final residual rows `R_rows`
    /// (device) and `next_tokens` (host: the token at cell+1), in batches through this path's GEMMs and its idle
    /// scratch - call it from on_chunk.  false with `err` empty: not applicable here (a ring or hybrid K/V, another
    /// device, too little scratch; STRATA_MTP_BATCH=0), the caller runs the drafter's own pass.  Not bit-identical to
    /// that pass (FP16 GEMMs instead of Q8_1 activations): the drafts may differ, never the target's tokens' logits.
    /// A resident destination requires its active on_segment lease: residual and next-token pointers, slot,
    /// interval and prompt length must match that owner. A harmless decline uses MtpDrafter::prefill_slot,
    /// never the ambient working draft state. Slot 8 retains the scalar on_chunk interface.
    /// `why` (optional): on a decline (false, `err` empty) the reason, a static string.
    bool draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                  std::string& err, const char** why = nullptr, int slot = 8, int64_t prompt_length = 0);
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
    /// `borrow_spill=true` is refused before allocation: its hybrid spill suffix has no admission price.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0,
              bool borrow_spill = false);

    /// With borrowed buffers: lay them out again for chunks of `chunk` tokens (at most `init`'s) in `borrow` - a
    /// request lends only the slots its prompt needs.  The stream must be idle (between prompts).
    bool relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err);
    int64_t chunk() const;
    /// Resolved allocation ownership, not permission to borrow. False before successful init.
    bool owns_work_buffers() const;
    bool bind_request(uint64_t request, std::string& err);
    /// Capture-only, enabled before init. Arm EVERY split stage with the same
    /// owner and whole run interval. Retains the final <=2047 absolute rows across
    /// chunks in owned mapped staging; no mixed/normed activations are exposed.
    /// Views are host-complete after successful run + chain drain, and borrowed
    /// until release/reset/destruction. No proposal or CaptureStore publication.
    bool enable_dflash2_capture(std::string& err);
    bool arm_dflash2_capture(core::dflash2::CaptureOwner owner, int64_t frontier, int64_t start, int64_t count,
                            std::string& err);
    bool dflash2_tap(core::dflash2::CaptureOwner owner, int tap_layer, core::dflash2::TapView& out,
                     std::string& err) const;
    bool release_dflash2_capture(core::dflash2::CaptureOwner owner, std::string& err);
    bool context_idle(std::string& err) const;

    void* stream() const;
    /// Last chunk's complete local residual output. Borrowed device storage,
    /// valid only after a successful full run/drain and until the next run.
    std::span<const float> stage_output() const;

    /// The share of the streamed experts' bytes DMA-able straight from pinned RAM (1 = all).  Sizes the streamed
    /// ring (a big one only pays when the copy engine, not the host copies, is the limit); set before bytes_needed.
    static void set_pinned_share(double share);
    static double pinned_share();
    /// The chunk size from which a chunk streams every expert the GPU does not hold, routed or not; a smaller one
    /// stages only the experts it routes to.  1024, or STRATA_PREFILL_STREAM_MIN, or the CPU share's limit once that
    /// is armed (3072 when STRATA_PREFILL_CPU_SHARE is set).
    static int64_t stream_all_min_tokens();
    /// --prefill-pipe (opt-in): the chunk a layer split's first stage reads an `n`-token prompt in, for `stages`
    /// stages, chunks of at most `cap` (the buffers' size) and a stage-chunk cost of b + a C with b/a = `ratio` tokens
    /// (0: unknown - STRATA_PREFILL_EQUAL's rule: the same chunk count as `cap` gives, evened out when its last chunk
    /// has stream_all_min tokens or more): the chunk count with the pipeline's least time, see the definition. `cap`
    /// when that is today's split or not 3% better than it. A function of its arguments only.
    static int64_t pipeline_chunk(int64_t n, int64_t cap, int stages, double ratio);
    /// --prefill-pipe for this run: 0 off (the buffers' chunks), 1 today's chunk count evened out (STRATA_PREFILL_EQUAL's
    /// rule), a number > 1 the rig's b/a in tokens (setup's --calibrate measures it). Default: STRATA_PREFILL_PIPE, else 0.
    static void set_pipe(double value);
    /// The same for the next prompts only (the `pipe=` request key, setup's calibration); < 0 = the run's own value.
    static void set_pipe_request(double value);
    /// The --prefill-pipe value in force (the request's, else the run's).
    static double pipe_setting();
    /// What a --prefill-pipe value plans by: -1 off, 0 today's chunk count evened out, else b/a in whole tokens (two
    /// spellings of one value, e.g. a request key and the flag, compare equal).
    static int64_t pipe_key(double value);
    /// The stages of the prompt path from this one on (1 without a layer split).
    int stage_count() const;
    /// --prefill-pipe on a layer split's first stage: the chunk an `n`-token segment is read in when it is read whole
    /// (pipeline_chunk over chunk(), the stage count and pipe_setting()), or 0 when the pipe is off here (one stage,
    /// value 0, or a later stage) and the buffers' chunk applies.  The engine plans a segment once, on its whole
    /// length, and reads every part of it with force_chunk, so its chunks do not depend on how it is cut into runs.
    int64_t plan_chunk(int64_t n) const;
    /// The next run on this (first) stage only: chunks of `chunk` tokens (at most chunk()) instead of a plan, from
    /// its first token; `single`: the segment is one chunk (the stage helper may bind); `planned`: the chunks came
    /// from a --prefill-pipe plan (such a read takes no CPU share: its chunks overlap on the stages by design).
    void force_chunk(int64_t chunk, bool single, bool planned);
    /// One log line for a planned segment (the same as a whole read logs when its plan differs from chunk()).
    static void log_plan(int64_t n, int64_t chunk, int stages);
    /// The expert blobs this stage and the later ones copied host -> device so far (PrefillStats::experts_streamed).
    int64_t chain_experts_streamed() const;
    /// #340: the streamed ring's slot count for chunks that stream every expert, instead of the pinned-share rule
    /// (0 = that rule). Set before any `bytes_needed`/`init` (both count the ring); STRATA_PREFILL_RING still wins.
    static void set_ring_override(int slots);
    /// #583: the auto chunk scan's byte-budget ring, for chunks above `small_max` (0.1.39's auto chunk: a prompt that
    /// fits it keeps 0.1.39's ring).  0 slots = none.  A layer split's set_ring_override and STRATA_PREFILL_RING win.
    static void set_ring_budget(int slots, int64_t small_max);

    /// Device bytes `init` needs for a chunk of `chunk` tokens (what a borrowed region must hold).  `src`: init gets an
    /// ExpertSource (carve picks the fused MoE layout only then).
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                 bool src = true);

    /// The same without the streamed ring: what the chunk's own buffers cost.  The auto chunk scan sizes the chunk
    /// first and hands the ring what the chunk leaves over, so it needs the chunk priced on its own.
    /// What `init` allocates when the prompt path OWNS its buffers: each cudaMalloc rounded up to a 2 MiB page and the
    /// ring as one allocation (the startup sizing of a cache without a loan).
    static uint64_t bytes_needed_owned(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                       bool src = true);
    static uint64_t bytes_needed_no_ring(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                         bool src = true);

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
    /// is a resolved ring count (as returned by ring_slots_for), with the staged walk's minimum of 32 slots added
    /// by the same rule carve uses. 0 omits all staging slots for bytes_needed_no_ring; it is not a runnable layout.
    /// `src`: the ExpertSource the caller will pass to init (carve picks the fused MoE layout only then).
    static uint64_t bytes_needed_from(const core::ModelGeometry& g, const PrefillFacts& f, int64_t chunk,
                                      int ring_slots_override = -1, uint64_t ring_blob = 0,
                                      int stage_own_override = -1, bool owned_pages = false, bool src = true);
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

    /// STRATA_MMQ_DIRECT: the guard each sized expert-cache slot keeps after its blob (ExpertCache::set_slot_guard),
    /// so the prompt MMQ can read a resident expert in its slot.  0 unless STRATA_MMQ_DIRECT is 1 or 2 on a CUDA
    /// build with the pointer-table MMQ (STRATA_MMQ_TAB).
    static size_t slot_guard_bytes();

    /// Positions [pos0, pos0 + n) holding `tokens`; `ss.ple_prev` must be the two tokens before pos0 (oldest
    /// first, -1 for none) and is advanced to the last two of these.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    /// Same arithmetic as run(), with disjoint owner intervals in one union-row chunk. The command predicate
    /// is checked only before stage 0 issues a pack; pending commands drain the issued chain and return
    /// certified owner frontiers, not cancellation. No callback or descriptor outlives this call.
    /// Owner request/incarnation/physical session mappings must have been registered by bind_owner before
    /// preparation writes. Each owner's next_tokens certifies the residual/position+1-token pair for MTP.
    SegmentedResult run_segments(const OwnerSpan* owners, int count, std::string& err,
                                 const std::function<bool()>& command_pending);
    bool bind_owner(int slot, uint64_t request, uint64_t incarnation,
                    const std::array<core::SessionState*, 4>& sessions, std::string& err);
    // Read-only drained owner/operation witness; no rebind, upload or allocation.
    bool validate_owner(int slot, uint64_t request, uint64_t incarnation,
                        const std::array<core::SessionState*, 4>& sessions,
                        const std::array<uint64_t, 4>& allocations, uint64_t operation, std::string& err) const;
    std::function<bool(const OwnerSpan&, size_t stage, uint64_t execution, int64_t frontier,
                       std::string& err)> on_segment_stage;
    std::function<bool(Prefill&, const OwnerSpan&, const float* residual, std::string& err)> on_segment;

    /// One arithmetic chunk, with the live stack/arena retained at layer boundaries.
    /// step runs on the caller thread before each stage and after each layer's
    /// compute/copy completion. It must return with that device's decode work
    /// drained; other devices may retain verifier leases. No owner may rebind,
    /// relayout or recursively run this prompt while step is executing.
    /// Stage handoffs are serial in this experiment; no second bulk arena exists.
    bool run_retained(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err,
                      const std::function<bool(Prefill*, int64_t, std::string&)>& step,
                      bool layer_yields = true);

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
    /// The first stage only, checked before every chunk but the first: true ends the run there as if the prompt
    /// ended there.  `run` returns true once every stage has read the tokens before that chunk, and `stopped_at()`
    /// says how many of the run's tokens that is (-1 when the run read them all).  A read beside the batch slots
    /// (STRATA_PIPED_CONCURRENT) gives way to a waiting shorter request there.
    std::function<bool()> stop_at_chunk;
    int64_t stopped_at() const { return stopped_at_; }
    /// Grows this stage's (and the later stages') pinned upload slots to what a text chunk of `T` tokens uploads, so
    /// a run of such chunks allocates no pinned memory (STRATA_PIPED_CONCURRENT: an allocation or free on the read's
    /// thread takes the driver's lock).  Before the run, with nothing of this path in flight.  False: a slot could
    /// not be had (it then grows at first use, as before).
    bool reserve_uploads(int64_t T);

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
    /// (the CPU's own activation format). Null or explicit share 0: every expert on the GPU. Only for a pool no other
    /// thread runs meanwhile (no batch slots); on a layer split every stage may have it - one stage at a time takes it
    /// for a chunk. Set before `init`; resolve its eligibility with arm_cpu_share before admission.
    void set_cpu_pool(kernels::cpu::ExpertPool* pool);
    /// With STRATA_PREFILL_CPU_SHARE set and `applies` (a pool will be set): the chunks staged after their routing - the
    /// only ones the share applies to - go up to STRATA_PREFILL_CPU_SHARE_MAX tokens (default 3072) instead of 1024.
    /// Before allocation_config or any `bytes_needed` (the ring and small-chunk buffers follow the limit);
    /// `init` repeats the already-resolved choice. Startup fallback must not enable an unpriced pool/default.
    /// `by_default`: this path is one the share was measured on (CUDA, one GPU, no batch slots), so with the variable unset
    /// the share is on for chunks below 1024 tokens (STRATA_PREFILL_CPU_SHARE=0 turns it off).
    static void arm_cpu_share(bool applies, bool by_default = false);
    static bool cpu_share_enabled();

private:
    friend struct PrefillDrainFixture;
    static uint64_t bytes_needed_impl(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                      bool owned_pages, bool src);
    bool dflash2_enabled_ = false;
    kernels::cpu::ExpertPool* cpu_pool_ = nullptr;   ///< set_cpu_pool
    // Stage-1 pipeline: intermediate stages return after handing their chunk to
    // the direct successor. The public run() drains the chain once at prompt end.
    bool run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err,
                  const OwnerSegment* segments = nullptr, int segment_count = 0, uint64_t execution = 0,
                  const StagePacket* incoming = nullptr, uint64_t incoming_generation = 0);
    StagePacket::Outcome join_successor(std::string& err);
    StagePacket::Outcome drain_pipeline(std::string& err);
    const std::function<bool(Prefill*, int64_t, std::string&)>* retained_step_ = nullptr;
    bool retained_layer_yields_ = true;
    bool retained_boundary(int64_t next_layer, std::string& err, bool bulk_complete = true);
    bool local_idle(std::string& err) const;
    std::atomic<bool> public_running_{false};
    Prefill* segment_root_ = nullptr;
    SegmentedResult* segment_result_ = nullptr;
    const OwnerSpan* segment_inputs_ = nullptr;
    int segment_input_count_ = 0;

    int64_t stage_lb_ = 0, stage_le_ = -1;
    Prefill* next_ = nullptr;
    Prefill* helper_ = nullptr;         ///< set_stage_helper
    int64_t stopped_at_ = -1;           ///< stop_at_chunk ended the last run after this many tokens (-1: it read all)
    int64_t force_chunk_ = 0;           ///< force_chunk, for the next run (0: none)
    bool force_single_ = false, force_planned_ = false;
    bool bind_stage_helper(int64_t T);  // binds the helper's buffers for a one-chunk prompt of T tokens
    /// FORK (HET-022): allocate one resident owner's prompt helper (streams, events, compact MMQ buffers, mapped
    /// host crossings) on its device and append it to the engaged set; false with `err` set when it cannot serve.
    bool add_prompt_helper(const PromptHelper& owner, int64_t cap_rows, std::string& err);
    StagePacket* next_packet_ = nullptr; ///< owns the direct successor's completion and its borrowed payload
    int hand_buf_ = 0;

    bool carve(std::size_t T, void* alloc);   // the device buffers of a chunk (prefill.cpp's Alloc)
    void release();                          // the destructor's cleanup (also `reset`'s)
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PrefillStats stats_;
};

}  // namespace strata::prefill
