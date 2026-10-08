// include/strata/core/verify.hpp - plan v0.3 P6: the speculative VERIFY window.
//
// T tokens at consecutive positions p0 .. p0+T-1 - the last accepted token and T-1 drafts - go through all 48
// layers in ONE captured graph, and the head's argmax is produced for every one of them.  Token t's argmax is
// what plain greedy decode would produce after token t, BIT FOR BIT: every kernel here is either the single-token
// kernel applied per token, or a multi-token kernel whose per-token arithmetic is the single-token kernel's
// (multi-column MMVQ in exact mode, the T-token GDN kernels, the per-token hit activation, the multi-token CPU
// expert rows).  So a draft is accepted exactly when greedy decode would have produced it.
//
// What the window costs is the dense weights read ONCE for T tokens and the union of the T tokens' missed
// experts on the CPU (measured on decode traces: 1.75x one token's misses for T=2, 2.4x for 3, 3.05x for 4).
//
// STATE.  The window appends K/V and indexer keys for all T positions and leaves the GDN state untouched.
// `commit(n_keep)` then makes the first `n_keep` tokens permanent: the GDN conv history and recurrent state are
// advanced by replaying those tokens from inputs the window stored, the indexer's key tail is restored from a
// snapshot and the accepted keys re-appended (a rejected key can land in a slot the current block still needs),
// and the PLE history is set to its snapshot after token n_keep-1.  K/V cells past the accepted prefix are simply
// overwritten when those positions are processed again, before any query can read them.
//
// Requires the default native decode configuration (native projections, fused GR, fused GDN, fast attention and
// selection, native indexer) and a profile-filled VRAM expert tier with its residency table on the device.
#pragma once

#include <cstdio>
#include <array>

#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/verify_ownership.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <map>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;
class RemoteExpertOpt;

/// The CPU pool for a window: x_f (n_tok, n_embd), ids (n_tok, k) -> out (n_tok * k, n_embd), hit rows zeroed.
/// A zero-row layer=-1 call acknowledges successful final consumption; -2 acknowledges poisoned graph drain.
using PoolMultiFn = bool (*)(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                             int64_t layer, const int64_t* positions, const int* request_slots, int group, int lane);

struct VerifyHits {
    const int32_t* d_res = nullptr;      ///< device [n_layers * n_expert] slot or -1
    const int32_t* h_res = nullptr;      ///< host [n_layers * n_expert] slot or -1 (when all >= 0 in stage: zero-doorbell)
    const uint8_t* cache_base = nullptr; ///< slot 0 of the VRAM expert arena
    const uint64_t* slot_off = nullptr;   ///< E-6: host per-slot offsets when slots differ in size (null: slot * blob)
    int64_t n_slots = 0;                  ///< E-6: how many (for the device copy)
    int64_t blob = 0;
};

/// Explicit request-owned rows; contiguous rows may be speculative positions of the same request.
/// Cancellation is linearized by the single host submitter before launch; a launched window must
/// commit/drain before another thread's cancellation can be observed.
enum class BatchRowMask : uint8_t { active, padding, finished, cancelled };
struct BatchRow {
    uint64_t request_id = 0, cancellation_generation = 0;
    SessionState* state = nullptr;
    int slot = -1;
    int32_t token = -1;
    int64_t position = 0, sequence_begin = 0, sequence_end = 0;
    int verification_span = 1, output_offset = 0;
    BatchRowMask mask = BatchRowMask::padding;
};

struct BatchResources {
    uint64_t device_bytes = 0, pinned_bytes = 0, history_bytes = 0;
    size_t graph_keys = 0, graph_execs = 0;
};

class Verifier {
public:
    Verifier() = default;
    ~Verifier();
    Verifier(const Verifier&) = delete;
    Verifier& operator=(const Verifier&) = delete;
    void set_remote_expert_opt(RemoteExpertOpt* opt) { remote_opt_ = opt; } // before init/capture

    /// The watchdog's view of the window in flight (issue #31): the layer, the GPU's sequence, the flags.
    void diag(std::FILE* f) const;
    /// #649 (STRATA_VERIFY_TRACE=1, opt-in): the last windows' host-side handshake events and the GPU's breadcrumbs
    /// (in mapped memory, so they read while the GPU hangs).  Nothing when the trace is off.
    void trace_dump(std::FILE* f) const;

    /// #267: raise every flag the window's spin kernels wait on past any ring (UINT32_MAX), so a window the GPU
    /// cannot finish drains instead of staying resident, then wait up to `timeout_ms` for its streams.  For the
    /// paths that give up on the engine (a timed-out window, the serve watchdog): the window then ran on whatever
    /// the flags guarded, so this verifier refuses every later window.  True when the streams finished.
    bool release_gpu_waits(int timeout_ms);

    /// `max_t` <= kVerifyMaxT.  `head` may be null (the canonical head is then run per token).
    bool init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
              const NativeHead* head, int max_t, std::string& err);
    /// Device arena only; same allocation sequence as init, no allocation or GPU work.
    static uint64_t planned_device_bytes(const ModelGeometry& g, int64_t context, int64_t vocab, int max_t);

    /// Boot/calibration only: capture/upload every width 1..max_t and commit, without launch
    /// or state mutation. Set all capture-affecting configuration first. Chained stages follow
    /// on their own devices. Keep a denying GraphCaptureObservationScope while serving to
    /// prohibit unseen captures; observed preparation is not an opaque allocator hard bound.
    bool prepare_graphs(std::string& err);
    /// Distinguishes independently in-flight verifier stages sharing one dispatch.
    void set_worker_lane(int lane) { worker_lane_ = lane; }

    /// All APIs have ONE host submitter. Reject overlapping/uncommitted windows;
    /// watchdog release poisons the verifier permanently. Bind a nonzero caller
    /// ID at every serving request start, before touching request state/staging.
    /// Legacy non-serving callers may leave it zero: staging is still checked,
    /// but diagnostics label that identity unbound, never a request-ID proof.
    bool set_request_id(uint64_t request_id, std::string& err);
    /// One window: tokens[0..T) at consecutive positions, out[t] = head pick.
    /// The PLE rows are gathered here from `ss.ple_prev` and the tokens.  Captures the T-token graph on first use.
    bool run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out, std::string& err);
    /// Diagnostics: row `t` of the last window's head logits (n_vocab floats) to the host. Valid after run().
    bool copy_logits(int t, float* host) const;
    int64_t vocab() const { return next_ ? next_->vocab() : n_vocab_; }
    /// The sampling the verify window's head applies (temperature / top_p / top_k / seed).  Set per
    /// request; greedy by default.  The sampling itself runs OUTSIDE the captured graph - its
    /// parameters would otherwise be baked forever - so this can change between requests freely.
    void set_sampling(const strata::kernels::SamplerParams& sp) {
        sampling_ = sp;   // row t of a window at pos0 draws Philox(seed, pos0 + t): see run()
        if (next_) next_->set_sampling(sp);
    }

    /// The penalty histories for `sampling_.penalty_last_n`: ONE ROW PER WINDOW ROW, T rows of `history_len`
    /// int32 slots at that stride (`strata::kernels::penalty_rows` builds them), most recent token LAST, unused
    /// front slots -1 (the kernel reads only the tail window).  Row t follows the window's drafts 1..t - staging
    /// row 0 alone (before 0.1.19) left the drafted rows with unwritten histories.  Null disables the penalties
    /// entirely - the neutral run's sampling call is byte-for-byte what it was.  The engine re-uploads the rows
    /// before every window; the buffer must hold kVerifyMaxT rows and stay alive across the request.
    void set_history(const int32_t* history, int history_len) {
        hist_d_ = history;
        hist_len_ = history_len;
        if (next_) next_->set_history(history, history_len);
    }
    /// Off: `run` skips the request's head sampling and `out` is the recorded greedy pick.  For windows whose
    /// picks are discarded - a prompt read through windows commits every token - so they cost no sampler launch
    /// or sync and never read a history staged for another position.
    void set_head_sampling(bool on) { head_sampling_ = on; if (next_) next_->set_head_sampling(on); }

    /// LAYER SPLIT (multi-GPU): this verifier runs layers [layer_begin, layer_end) of every window.  A stage that
    /// does not start at layer 0 takes its residual from `handoff_in` instead of embedding the tokens; a stage that
    /// does not end at the last layer writes its residual to `handoff_out` and has no head.  The hand-off holds,
    /// per token, the residual R (hc x n_embd), the last layer's pending write bo (n_embd) and inject (hc): the next
    /// stage folds that write into its first read exactly as the unsplit window does, so the split is bit-exact.
    /// Both pointers must be device-visible (mapped pinned memory, portable when the stages are on two devices).
    /// Set before `init`.  Default: the whole model, no hand-off.
    void set_stage(int64_t layer_begin, int64_t layer_end, const float* handoff_in, float* handoff_out) {
        lb_ = layer_begin; le_ = layer_end; hand_in_ = handoff_in; hand_out_ = handoff_out;
    }
    /// The next stage: `run` and `commit` continue into it (its pool calls get `next_user`); sampling settings
    /// and `final_R` are the last stage's.
    void set_next(Verifier* next, void* next_user) { next_ = next; next_user_ = next_user; }
    /// floats per token in a hand-off buffer
    static int64_t handoff_floats(const ModelGeometry& g) { return (int64_t) g.hc * g.n_embd + g.n_embd + g.hc; }

    /// Keep the first `n_keep` (1..T) tokens of the last window; advances `ss.ple_prev` by them.
    bool commit(int n_keep, std::string& err);

    // ================================ SEVERAL SEQUENCES IN ONE WINDOW ================================
    //
    // A batch window holds S INDEPENDENT sequences, one token each: row s is slot s, at slot s's own position,
    // reading and writing slot s's own state (GDN recurrence and conv history, QSA K/V and indexer, PLE history),
    // which lives in `slots[s]` - a session carved like this verifier's own (same layer range, same max_cells).
    // Everything that is per row already (hyper-connections, dense projections, router, shared expert, the
    // routed experts on the GPU and the CPU, the head) runs once over the S rows, so the weights are read once
    // per window for all the sequences.  Row s's arithmetic is the single-token window's, so a slot's greedy
    // tokens are its solo greedy tokens (modulo the multi-token CPU kernel choice: STRATA_IQ_MT_MIN=1).
    //
    // Request-local sampling and penalties; no drafts (MTP) in a batch window. `init_slots` once after `init` (S <= max_t); a layer
    // split's stages each get their own sessions, and run_slots/commit_slots continue into the next stage.
    bool init_slots(const std::vector<SessionState*>& slots, std::string& err);
    int n_slots() const { return (int) slots_.size(); }
    /// One batch window over slots [0, S): tokens[s] at positions pos[s]; out[s] = the greedy pick after it.
    bool run_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err);
    /// Capture cache is capped at kVerifyBatchGraphKeys distinct row-order/base
    /// keys (window + commit per key). An unseen key past the cap is rejected
    /// before capture; graphs are never evicted while a request can use them.
    /// The same over the S slots `rows` (row t is slot rows[t], any distinct slots in any order): the slots not
    /// listed are not touched, so an idle slot keeps its state (a finished conversation it may continue later).
    bool run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user,
                       int32_t* out, std::string& err);
    /// Compact active metadata rows into run_slot_rows. Inert rows are never
    /// staged, sampled or committed and return -1 at their output offset.
    /// An all-inert/empty window still requires exactly one commit_slots call.
    bool run_slot_rows(const BatchRow* rows, int count, PoolMultiFn pool, void* user,
                       int32_t* out, std::string& err);
    uint64_t slot_generation(int slot) const {
        return slot >= 0 && slot < (int) slots_.size() ? slot_request_id_[slot] : 0;
    }
    /// Safe only between operations. Releases graph variants or all batch-only
    /// scratch without touching resident/cached state, solo graphs or MTP borrowers.
    /// init_slots may bind the same stable sessions again after resource release.
    bool release_batch_graphs(std::string& err);
    bool release_batch_resources(std::string& err);
    BatchResources batch_resources() const;
    /// Keep every row of the last batch window: 
    bool commit_slots(std::string& err);
    /// Commit an accepted prefix in each contiguous slot group of the last window. `keep` has
    /// one entry per slot (indexed by slot ID), each in 0..that slot's group length; zero discards cancellation.
    bool commit_slot_prefixes(const int* keep, std::string& err);
    /// An operator may tighten the priced 32-key bound, never enlarge it or request an unbounded cache.
    void set_batch_graph_limit(size_t n) { batch_graph_limit_ = n == 0 ? kVerifyBatchGraphKeys : std::min<size_t>(n, kVerifyBatchGraphKeys); }

    // ---- The stages of a layer split as a PIPELINE. A window carries only the active `rows` of a group.
    // It launches on ONE stage with its commit immediately behind it; the host serves every stage's rings
    // from one thread (batch_poll does not block). Packed row t uses hand-off row base+t on every stage.
    // Group hand-off ranges remain disjoint, and inactive resident state is neither read nor committed.
    bool batch_launch(int base, const int* rows, int S, const int32_t* tokens, const int64_t* pos, std::string& err);
    /// 1 = this stage's window and commit are done (the last stage's picks are in batch_out), 0 = still running,
    /// -1 = an error (err).  Serves every layer that has rung so far.
    int batch_poll(PoolMultiFn pool, void* user, std::string& err);
    bool batch_busy() const { return b_running_; }
    /// Resident lifecycle preflight. A pending solo commit is drained; owned or
    /// poisoned scratch refuses before any resident state is copied or rebound.
    bool context_idle(std::string& err);
    /// Completion may retire a finished pipeline group while an unrelated group
    /// runs. It must not publish reuse while this slot still has a consumer.
    bool slot_consumers_idle(int slot, bool expert_pending, std::string& err);
    /// Bind/rebind an idle resident slot at admission. IDs must be nonzero and
    /// distinct across slots. History stays alive until rebind; do not mutate it
    /// while an operation owns scratch. Also forwards to chained stages.
    bool set_slot_context(int slot, uint64_t request_id, const strata::kernels::SamplerParams& sp,
                          const std::vector<int32_t>* history, bool steering, std::string& err);
    /// Last GPU-validated owner, zero before validation/rebind or after failure.
    uint64_t validated_slot_request_id(int slot) const {
        return !released_.load() && slot >= 0 && slot < (int) slots_.size() ? validated_slot_id_[slot] : 0;
    }
    const int32_t* batch_out() const { return b_out_; }
    bool last_stage() const { return g_ != nullptr && le_ == g_->n_layers; }
    /// commit() returns without waiting for its graph (a single-GPU session sets it): the next window follows it on
    /// the same stream and the drafter reads nothing it writes, so it overlaps the draft. Whoever reads or writes
    /// the session from another stream or the host afterwards (a new request, a checkpoint, a snapshot, the prompt
    /// path, the end of a run) calls wait_commit() first.  STRATA_COMMIT_SYNC=1 keeps the wait.
    static void set_commit_async(bool on);
    /// Waits for the last commit graph when commit() did not (an event recorded after it, not the whole device);
    /// false with `err` when it failed.  Free when nothing is pending.
    bool wait_commit(std::string& err);

    // ---- PIPELINED WINDOWS (--pipeline-windows, a layer split on two GPUs).  One conversation's windows with the
    // stages overlapped: stage 0 runs window K+1 while stage 1 still runs window K.  The same window as `run`, driven
    // without blocking the host, so one host thread keeps a window in flight on each stage (the batch pipeline's
    // pattern, batch_launch / batch_poll, for one sequence with drafts).  Two verifiers per stage (one per window
    // parity) share the stage's stream and each has its own hand-off.  `pl_launch` stages and launches (it never
    // captures: `capture_all` first, with nothing in flight), `service` serves the layers whose doorbells have rung
    // and returns at once, `done` polls the window's completion, `pl_finish` reads the picks, `pl_commit_async`
    // queues the commit.  Nothing chains to `next_`: the caller drives every stage.
    /// The compute stream to use instead of a private one (the two verifiers of one stage share it).  Before `init`.
    void set_stream(cudaStream_t s) { ext_stream_ = s; }
    /// Every layer copies the token rows to the host (doorbell_publish), not only the layers with a routed expert
    /// outside the VRAM tier by the device's residency table: with windows in flight the adaptive tier marks an
    /// expert evicted on the host (the pool then computes it on the CPU) before the device table follows.  Also turns
    /// the device-planned layers (E-6) off.  Before `init`.
    void set_always_publish(bool on) { always_publish_ = on; }
    cudaStream_t stream() const { return cs_; }
    int device() const { return device_; }
    /// Capture every window size and the commit graph now (a capture syncs the stream: never with a window in flight).
    bool capture_all(std::string& err);
    bool pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string& err);
    /// Stage a window ahead of its launch (positions, and the PLE rows from `ple_prev` = the two tokens before the
    /// window as they WILL be, their pages prefetched).  A later `pl_launch` of the same window (T, pos0, tokens, and
    /// `ss.ple_prev` equal to `ple_prev` by then) skips the staging; anything else stages again.
    bool prestage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err);
    /// 1: every layer served; 0: the GPU has not reached the next layer yet; -1: an error (`err`).
    int service(PoolMultiFn pool, void* user, std::string& err);
    bool in_flight() const { return fl_active_; }
    /// The window's graph (and its profile copy) completed; false while it runs.  An error sets `err`.
    bool done(std::string& err);
    /// After `done`: the profile, the last stage's host sampling and picks (`out` may be null on an earlier stage).
    bool pl_finish(int32_t* out, std::string& err);
    /// The commit without a host sync; `ss.ple_prev` advances now (host side).  A second call for the same window
    /// (after its state was restored) replays it with another count.
    bool pl_commit_async(int n_keep, std::string& err);
    /// Fold another verifier's counters and GPU profile into this one's (the two verifiers of one stage report once).
    void absorb_stats(Verifier& o);
    /// The watchdog's line for a pipelined verifier: in flight, layers served, the GPU's ring and flags, its events.
    void diag_pipelined(std::FILE* f, const char* name) const;

    /// Measurement hook (STRATA_LOGPOS): after run(), write one line per row t of the last window's head -
    /// "pos target logprob top top_logprob hit extra_logprob target_logprob_without_extra" - where row t is the
    /// distribution at pos0 + t, targets[t] is the token at pos0 + t + 1, extra_logprob is the log-probability of
    /// `extra_id` in the same row, and the last column is the target's log-probability in that row renormalized
    /// over every token but `extra_id` (both nan when they do not apply).  A layer split's earlier stage forwards
    /// to the stage that holds the head.
    bool window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                         std::string& err);

    /// Token t's residual after the last layer, (hc, n_embd) on the device, valid until the next `run`.
    const float* final_R(int t) const;
    const float* final_R_all() const { return next_ ? next_->final_R_all() : R_; }

    /// The head's vocabulary size (the row width `read_logits_rows` copies).  The window graph computes the
    /// head's FULL rows for every one of its T positions (`head_logits_`, T x n_vocab); the decode path reads
    /// only the argmax, so the rows themselves were unreadable until #58 needed them for `--dump-logits`.
    int64_t n_vocab() const { return n_vocab_; }
    /// Rows [t0, t1) of the last `run`'s head (n_vocab floats each) copied to `dst` (host), the stream synced
    /// on return.  Valid from `run` until the next `run` or `commit` of a new window - `commit` itself leaves
    /// the rows untouched, so the caller reads the round's committed rows after `commit`.  A layer split's
    /// rows live in its last stage, so the call chains there.  Dump-only: the decode path never reads rows.
    bool read_logits_rows(int t0, int t1, float* dst, std::string& err);

    /// The GPU plan the pool writes each layer (VRAM hits + the PCIe share of the misses); give it to the
    /// dispatch (`ExpertDispatch::plan`) before the first `run`.
    GpuPlanSink* plan_sink() { return &sink_; }
    /// Plan v0.3 P6: split the window into two token groups and pipeline the CPU experts of one with the GPU work
    /// of the other (default on).  Set before the first `run`.
    void set_split(bool on) { split_ = on; }
    /// Plan v0.3 P6: how the PCIe share of the misses reaches the GPU: 0 = DMA into staging (the copy engine works
    /// beside the CPU; best when the CPU is compute-bound, the i-quants), 1 = the grouped kernel reads the mapped
    /// arena directly, 2 = a copy kernel stages it inside the graph (no API calls on the pool's thread; best when
    /// the CPU is RAM-bound, Q2_0).  Set before the first `run`.
    void set_pcie_mode(int mode) { sink_.pcie_mode = mode; }
    /// the pool never plans a PCIe share (--pcie-frac 0): the window skips that path.  Before the first run.

    double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
    int64_t windows = 0;
    /// STRATA_VERIFY_PROFILE=1 - GPU stage times of the windows since the last call (ms per
    /// window), as one line; empty when off.
    std::string profile_report();

private:
    RemoteExpertOpt* remote_opt_ = nullptr;
    bool capture(int T, std::string& err);
    // batch windows (see init_slots)
    std::vector<SessionState*> slots_;
    bool batch_rec_ = false;               ///< record_window is capturing a batch window
    int row_base_ = 0;                     ///< ... its hand-off rows start here (a pipeline group's own rows)
    int brow_[8] = {};                     ///< ... and row t is slot brow_[t]
    bool last_batch_ = false;              ///< the last run was a batch window (set_plan_slot: one group)
    using BatchKey = std::array<int, 11>;
    std::map<BatchKey, cudaGraphExec_t> exec_bm_, commit_bm_;
    size_t batch_graph_limit_ = kVerifyBatchGraphKeys;
    int last_rows_[8] = {};                ///< the slots of the last batch window's rows
    BatchKey bkey(const int* rows, int S, int hbase) const {
        BatchKey key{};
        key[0] = hbase; key[1] = S; key[2] = ar_off_ ? 1 : 0;
        for (int t = 0; t < S; ++t) key[3 + t] = rows[t];
        return key;
    }
    // batch_launch / batch_poll
    bool b_running_ = false;
    int64_t b_k_ = 0, b_steps_ = 0;
    std::chrono::steady_clock::time_point b_last_;
    int32_t b_out_[8] = {};
    std::vector<strata::kernels::SamplerParams> slot_sp_;
    const std::vector<int32_t>* slot_history_[8] = {};
    int slot_steering_[8] = {};
    int32_t* steering_b_ = nullptr;
    int32_t* history_b_ = nullptr;
    std::vector<int32_t> history_stage_b_;
    int history_len_b_[8] = {};
    VerifyScratchOwnership scratch_;
    // On every exit after acquisition, either transfer ownership explicitly or
    // drain/poison launched work. Unlaunched setup failures are safely retryable.
    struct ScratchGuard {
        Verifier& v;
        bool launched = false;
        bool keep = false;
        ~ScratchGuard();
    };
    bool acquire_scratch(std::string& err);
    bool fail_operation(std::string& err);
    bool reserve_batch_key(const BatchKey& key, std::string& err);
    bool finish_self_commit(std::string& err);
    static uint64_t layout_arena(const ModelGeometry& g, int64_t context, int64_t vocab, int max_t, Verifier* owner);
    bool observe_committed_rows(int count, std::string& err, const int* keep = nullptr);
    void* h_route_trace_ = nullptr;
    void* m_route_trace_ = nullptr;
    uint64_t observed_rows_ = 0;
    uint64_t observation_limit_ = 256; // opt-in extended lifecycle vehicle: at most 544, still 16 rows/file
    std::FILE* row_trace_ = nullptr;
    uint64_t request_id_ = 0, canary_epoch_ = 0;
    uint64_t slot_request_id_[8] = {};
    uint64_t validated_slot_id_[8] = {};
    uint64_t batch_device_bytes_ = 0, batch_pinned_bytes_ = 0;
    void report_batch_resources(const char* phase) const;
    VerifyCanaryInput expected_canary_[8] = {};
    VerifyCanaryInput *h_canary_ = nullptr, *m_canary_ = nullptr;
    VerifyCanaryOwner *h_owner_ = nullptr, *m_owner_ = nullptr;
    VerifyCanaryOwner* owner_ = nullptr;
    VerifyCanaryOutput *canary_ = nullptr, *h_canary_out_ = nullptr, *m_canary_out_ = nullptr;
    VerifyCanaryOutput *h_canary_commit_ = nullptr, *m_canary_commit_ = nullptr;
    void stage_canaries(const int* rows, int T, const int32_t* tokens, const int64_t* pos, int64_t pos0);
    void record_canaries(int T, const int* rows, bool begin, bool commit, cudaStream_t stream);
    bool check_canaries(bool commit, std::string& err);
    bool validate_slot_context(int slot, std::string& err);
    bool sample_rows(int S, std::string& err);   ///< the sampled slots' rows of the last batch window
    int32_t* h_commitb_ = nullptr; int32_t* m_commitb_ = nullptr;   // per slot [1, 0, pos, -1 ..], stride 2 + max_t
    int32_t* commitb_ = nullptr;
    float* tail_snap_b_ = nullptr;         ///< per (slot, QSA layer) indexer tail snapshot
    void* arena_b_ = nullptr;
    int64_t last_pos_b_[8] = {};
    bool capture_batch(const int* rows, int S, int hbase, std::string& err);
    void collect_profile();   ///< STRATA_VERIFY_PROFILE: add the last window's stamps to prof_sum_
    void accumulate_profile(const unsigned long long* stamps);   ///< one window's stamps (host copy) into prof_sum_
    // pipelined windows (pl_launch ...)
    cudaStream_t ext_stream_ = nullptr;   ///< set_stream: the stage's shared stream (not destroyed here)
    bool always_publish_ = false;
    void pl_stage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2]);
    cudaEvent_t ev_done_ = nullptr, ev_commit_ = nullptr;
    unsigned long long* prof_pin_ = nullptr;   ///< pinned host copy of the stamps (pipelined windows)
    bool fl_active_ = false, fl_prof_ = false, fl_ple_ = false, commit_live_ = false, pl_prestaged_ = false;
    int fl_T_ = 0;
    int64_t fl_k_ = 0, fl_total_ = 0;
    double fl_since_ms_ = 0, fl_flush_ms_ = 0, fl_launch_ms_ = 0;
    PoolMultiFn fl_pool_ = nullptr;
    void* fl_user_ = nullptr;
    int32_t pl_prev_[2] = {-1, -1};
    std::vector<uint32_t> pl_ple_rows_;   ///< the window's PLE rows (T x PLE_N_HEADS), gathered when layer 0 is served
    bool capture_commit_batch(const int* rows, int S, int hbase, std::string& err);
    bool stage_batch(const int* rows, int S, int hbase, const int32_t* tokens, const int64_t* pos, std::string& err);
    strata::kernels::SamplerParams sampling_ = [] {
        strata::kernels::SamplerParams s;
        s.greedy = true;
        s.temperature = 0.0f;
        return s;
    }();   ///< greedy by default; per-request via set_sampling
    const int32_t* hist_d_ = nullptr;   ///< penalty-history row (set_history); null = no penalties apply
    int hist_len_ = 0;
    bool head_sampling_ = true;          ///< set_head_sampling
    int device_ = -1;                    ///< the device `init` ran on: run/commit switch to it (layer split)
    std::atomic<bool> released_{false};  ///< #267: release_gpu_waits ran (maybe on the watchdog thread): no more windows
    bool all_resident_ = false;           ///< 100% of experts in [lb_, le_) resident in VRAM: zero-doorbell graph
    /// #871: the zero-doorbell graph plans from the device residency table alone, so it is only right while every
    /// expert of the stage is in VRAM.  A prompt loan, a VRAM shrink or an adaptive swap marks some -1 for a while:
    /// every window then runs the doorbell graph (exec_nr_ / the batch key's top bit), the same as a stage that never
    /// was all-resident.  refresh_ar() looks at the host table before each window.
    bool ar_off_ = false;
    bool ar_on() const { return all_resident_ && !ar_off_; }
    void refresh_ar();
    uint32_t* h_plan_err_ = nullptr; uint32_t* m_plan_err_ = nullptr;   // set by resident_plan on a -1 (all-resident graph)
    const int32_t* h_res_ = nullptr;      ///< the host residency table (VerifyHits::h_res)
    bool device_plan_ = false;            ///< E-6: resident-only layers planned on the device (STRATA_VERIFY_DEVICE_PLAN)
    uint32_t* skip_ = nullptr;            ///< E-6: per group, the ring whose plan the device built (0: the host's)
    unsigned* qcnt_ = nullptr;             ///< S26 STRATA_QFUSE: the HC read's q8_1 group counters (n_embd / 32)
    unsigned long long* slot_off_d_ = nullptr;   ///< E-6: the slot offsets on the device
    int64_t lb_ = 0, le_ = -1;           ///< set_stage: the layers this verifier runs (-1: to the last)
    const float* hand_in_ = nullptr;
    float* hand_out_ = nullptr;
    Verifier* next_ = nullptr;
    void* next_user_ = nullptr;
    bool ple_stage() const { return lb_ <= 1 && 1 < le_; }   ///< holds layer 1, where the PLE block runs
    void stage_inputs(int T, const int32_t* tokens, int64_t pos0);
    bool copy_used_ = false;
    bool capture_commit(std::string& err);
    bool record_window(int T, cudaStream_t cs, std::string& err);
    // #649: STRATA_VERIFY_TRACE=1 - a host event ring (trace_ev) and GPU breadcrumbs: the profiler's stamp points,
    // per layer and token group, written to mapped memory (null when the trace is off)
    void trace_ev(const char* what, int64_t step, int64_t layer, int64_t aux) const;
    unsigned long long* trace_h_ = nullptr;
    unsigned long long* trace_m_ = nullptr;
    size_t trace_n_ = 0;
    static constexpr int kProfPer = 33;              // stamps per layer (32 left the hc-read second
                                      // half's up-stamp at slot 32 = the next layer's slot 0: D8)
    bool prof_on_ = false;
    unsigned long long* prof_ = nullptr;              // device: n_layers * kProfPer + 4 stamps
    std::vector<unsigned long long> prof_h_;
    double prof_sum_[2][kProfPer] = {};   // [GDN / QSA layers][stage]
    int64_t prof_windows_ = 0;

    const WeightTable* wt_ = nullptr;
    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    VerifyHits hits_;
    const NativeHead* head_ = nullptr;
    int max_t_ = 0;
    float* ple_key_ = nullptr;   ///< STRATA_PLE_BATCH: the window rows' PLE key / value projections
    float* ple_val_ = nullptr;
    int last_t_ = 0;
    int64_t last_pos0_ = 0;
    int32_t last_tokens_[8] = {};
    int64_t n_vocab_ = 0;
    cudaStream_t cs_ = nullptr;
    cudaStream_t sh_cs_ = nullptr;
    cudaEvent_t ev_fork_ = nullptr, ev_join_ = nullptr;
    cudaGraphExec_t exec_[9] = {};
    cudaGraphExec_t exec_nr_[9] = {};   // #871: the doorbell variant of a stage that is all-resident otherwise
    cudaGraphExec_t commit_exec_ = nullptr;

    // mapped staging (host pointer, device alias)
    int32_t* h_tok_ = nullptr;   int32_t* m_tok_ = nullptr;     // T
    int32_t* h_step_ = nullptr;  int32_t* m_step_ = nullptr;    // T * kStepCount
    int32_t* h_pos_ = nullptr;   int32_t* m_pos_ = nullptr;     // T * n_head
    int32_t* h_commit_ = nullptr; int32_t* m_commit_ = nullptr; // [n_keep, n_keep-1, pos_0 .. pos_{T-1}]
    float* h_ple_ = nullptr;     float* m_ple_ = nullptr;       // T * n_embd
    int32_t* h_out_ = nullptr;   int32_t* m_out_ = nullptr;     // T argmax ids
    float* h_x_ = nullptr;       float* m_x_ = nullptr;         // doorbell payload: T * n_embd
    int32_t* h_ids_ = nullptr;   int32_t* m_ids_ = nullptr;     // T * k
    float* h_w_ = nullptr;       float* m_w_ = nullptr;         // T * k
    uint32_t* h_seq_ = nullptr;  uint32_t* m_seq_ = nullptr;
    uint32_t* h_flag_ = nullptr; uint32_t* m_flag_ = nullptr;
    uint32_t* h_flagA_ = nullptr; uint32_t* m_flagA_ = nullptr;  // the GPU plan is in place
    uint32_t* h_flagB_ = nullptr; uint32_t* m_flagB_ = nullptr;  // the PCIe share's DMA copies have landed
    cudaEvent_t commit_done_ = nullptr;   // recorded after an async commit (set_commit_async); see wait_commit
    bool commit_pending_ = false;
    cudaStream_t copy_ = nullptr;                                 // the copy engine's stream (DMA of missed experts)
    struct FlagSet { uint32_t* flag; uint32_t value; };
    FlagSet flag_sets_[2 * 64 * 2] = {};                          // host-function arguments, one per (layer, group)
    static void fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes);
    static void raise_flag(uint32_t* flag, uint32_t value);
    int32_t* h_plan_ = nullptr;  int32_t* m_plan_ = nullptr;     // counts | start | dst | tok | ptr (as int32 pairs)
    int64_t plan_i32_ = 0;                                        // int32 words in the plan block
    GpuPlanSink sink_;
    int worker_lane_ = 0;
    int64_t worker_positions_[8] = {};
    uint32_t cur_layer_ = 0;
    static void publish_plan(void* ctx);
    void set_plan_slot(int grp);
    bool split_ = false;   // opt-in (--spec-split): exact but slower, see the overlap study
    int groups_[9] = {};
    float* h_ymiss_ = nullptr;   float* m_ymiss_ = nullptr;     // T * k * n_embd

    // device
    void* arena_ = nullptr;
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *commit_ = nullptr;
    float *ple_ = nullptr, *emb_ = nullptr, *R_ = nullptr, *mixed_ = nullptr, *bo_ = nullptr;
    float *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *xn_ = nullptr;
    uint8_t* xq_ = nullptr;                                   // T columns of q8_1
    uint8_t* sh_xq_ = nullptr;                                // T columns of q8_1 for shared expert branch
    float *qkv_L_ = nullptr, *h_L_ = nullptr, *gate_L_ = nullptr, *beta_L_ = nullptr;   // per GDN layer
    float *z_ = nullptr, *y_ = nullptr, *y_dummy_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *idx_raw_L_ = nullptr;
    float *qidx_ = nullptr, *scores_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr, *attn_scratch_ = nullptr;
    float* tail_snap_ = nullptr;                              // per QSA layer
    int32_t* sel_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *hit_out_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr;
    int32_t* plan_ = nullptr;                                     // device copy of the plan block
    uint8_t* staging_ = nullptr;                                  // VRAM slots for the PCIe share of the misses
    static constexpr int64_t kStagingBlobs = 16;
    static constexpr int64_t kPcieGroupRows = 4;                  // the PCIe call's groups side by side (of <= 16)
    uint8_t* hit_xq_ = nullptr;
    uint8_t* nat_xq_ = nullptr;   // plan v0.3 P6: q8_1 activations for a native pack's grouped experts
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float *head_mixed_ = nullptr, *head_inj_ = nullptr, *head_logits_ = nullptr;
    uint16_t* sh_bf16_ = nullptr;
    uint8_t* arg_scratch_ = nullptr;   ///< argmax_rows' partials and counters
    int32_t* one_ = nullptr;           ///< device {1}: the n_keep of a one-token window, which commits itself
    float *sh_gate_ = nullptr, *sh_up_ = nullptr, *sh_g_ = nullptr;
    float* hist_snap_ = nullptr;                              // T * NG_HIST * NG_HC_DIM
    int64_t cap_ = 0, max_blocks_ = 0, attn_scratch_floats_ = 0;
};

}  // namespace strata::core
