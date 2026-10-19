# Several requests at once (batch slots)

By default Strata serves **one request at a time**: the others wait in the server's queue. With `"parallel": N`
(the engine's `--batch N`, also spelled `--slots N`) the engine keeps up to N conversations open and decodes them
**together**: every verify window then carries one token of each conversation, so the dense weights, the shared
expert, the head and every routed expert two conversations share are read once per window for all of them.
Combined with a layer split across several GPUs and `--batch-groups`, the cards also work on different
conversations at the same time instead of waiting for each other.

It is opt-in and changes nothing when the options are absent (#465; the engine part is PR #559).

**Qwen3.6-35B-A3B** has the same option on its own engine path, `strata-q36` (`"parallel": N`, `--batch N`, `--batch-mtp`, up to 8
slots, the same engine protocol and the same server). This fork's `strata` refuses the family; its numbers, what a slot costs and
where setup recommends it are in [QWEN36.md](QWEN36.md).

## Turning it on

One GPU: add `"parallel": 2` to the model's config (`strata-<model>.json`) and restart, or run setup with
`--parallel 2`. Setup recommends it only where it does not cost speed (below); any number you ask for is kept as
asked, with a note when it is more than setup would recommend.

The engine admits at most `kVerifyMaxT` (8) resident slots in every mode; a larger
`--batch` request is held to that count with a diagnostic explaining that the fork's
resident-slot ownership is bounded by the verify-window cap.
The frontend's `PARALLEL_MAX=8` check only warns and passes the request through;
the engine is the admission authority, and `engine.batch` reports its admitted value.

```
"parallel": 2
```

With MTP (`--mtp` and `--spec`), `--batch-mtp` (in the config's `args`, or `STRATA_BATCH_MTP=1` in the server's
environment) lets each batch slot verify one MTP proposal per window. It is opt-in; without it the batch behaviour
described below is exactly the one without MTP. It needs VRAM per slot for the draft state and buffers, so check the
engine's free-memory log before using it on a smaller card. An incompatible explicit `--batch-mtp` request
**refuses startup**; it is not silently disabled. It needs `--serve`, loaded `--mtp --spec >= 2`,
distinct devices for a layer split and at least two slots, except that packed prefill permits
one resident slot because a packed owner holds the window itself. The synchronous split route also refuses helper expert caches.
Historical one-GPU RTX PRO 5000 runs measured +31% to +39% total throughput with 2 to 4 clients (a RX R9700 run too).

**Native layer-split ownership:** one MTP scratch/head/weight owner on the last stage serves every slot;
each slot has private draft K/V, not a separate drafter object.
`--batch-mtp` gives each admitted slot its own MTP proposal; it does not raise the admitted slot count.
A synchronous window verifies up to four main/proposal pairs in the eight-row quantum;
six/eight admitted slots rotate their proposals through successive windows. Select
`--batch-groups 1`, or `STRATA_BATCH_MTP_SERIAL=1` with no explicit groups, for synchronous split MTP.
The layer-split default otherwise resolves multiple pipelined groups when the slot count permits:
without `--batch-mtp-pipeline`, that combination refuses rather than dropping MTP or choosing serial execution.

`--batch-mtp --batch-mtp-pipeline --batch-groups 2` retains the fork's separate two-group route.
It admits **effective even slot counts 2, 4, 6 or 8**, on a distinct-device layer split, with no resident multiplexing.
This is the pipelined variant's own replay bound and does not change the uniform admission cap.
Requested counts above eight are held to eight before the even-count check; an effective
seven refuses. Runtime fit must still leave two supported groups. The pipeline's constrained
helper-cache exemption remains; optimized helpers, automatic helper counts and adaptive helper composition refuse.
These native ownership/retirement statements do not retire the SYCL backend's own per-slot drafters or graph cache.
The measurements below are historical results from their named configurations, not qualification of this merge.

Measured on a layer split of 2x RTX 3080 20 GB (220 W cap) with a Xeon E5-2696 v4, UD-Q4_K_XL, `"layer_split": "23"`,
`"parallel": 2`, `--spec 4`: two concurrent greedy decodes of 300 tokens, the two streams' tok/s added, median of
the measurements (10 rounds of two streams after each restart of the engine, 20 measurements per restart). The
engine was 0.1.41 with the rest of our production stack on it (the shared KV pool, and the adaptive expert tier
running in batch windows in the first two arms; the pipelined path of the third never adapts in windows), so this is
`--batch-mtp` on top of that, not this change alone on main.

| arm | aggregate tok/s | restarts, measurements | per window |
| --- | --- | --- | --- |
| `--batch-mtp`, one group | **89.2** | 3, 60 | 3.3 rows, 34.6 ms, 66-67% of the proposals accepted |
| no `--batch-mtp`, one serial group | 77.3 | 2, 40 | 2.0 rows, 23.7 ms |
| no `--batch-mtp`, `--batch-groups` unset (the 0.1.41 default: two pipelined groups of one slot) | 77.7 | 1, 20 | one row per group step |

That is +15.4% for `--batch-mtp` against serial windows without it (95% bootstrap interval of the ratio of medians
+12.5% to +17.9%) and +14.8% against the pipelined default; the two arms without it tie. One stream alone decodes the
same with and without it (81.7 and 82.0 tok/s), and the two slot drafters cost 35 to 110 expert-cache slots on the
last GPU (50 MiB of private state each).

The pipelined default is better in one place: a new prompt of about 48k tokens arriving while one stream decodes was
read at 2508 tok/s with pipelined groups and 1217 tok/s in serial windows, and the decoding stream finished sooner
(25.0 against 18.0 tok/s over its whole run); `--batch-mtp` does not change that. So it pays where concurrent decodes
dominate; with long reads arriving beside decodes the pipelined groups can be the better choice. Not measured: more
than two stages (the report on #1253 found no gain at four stages), more than two slots, sampled decoding, an
upstream-main binary (the pipelined arm is our build, started with the flags 0.1.41 resolves to).

Our measurement of `--batch-mtp` on a layer split with 2x R9700 and all experts in VRAM: 80.2 -> 62.7 tok/s (-21.9%, 0/5 pairs faster) against the pipelined default, so it pays only when the experts do not all fit in VRAM.

Earlier, on engine 0.1.40.3 with this change and the same two cards, two concurrent streams: UD-Q4_K_XL 39.9 to 43.5
tok/s without and 44.5 to 51.3 with `--batch-mtp` (2 runs per arm), and a small Coder IQ1_M test model with all
experts in VRAM 84.0 without and 101.2 with it, the greedy text of both streams identical.

Text: with the same expert cache on both arms (the slot drafters take VRAM, so the cache differs unless the
reserve is adjusted) and `--pcie-frac 0`, `--adapt-every 0`, the greedy text with `--batch-mtp` was equal to the text
without it for 5 short prompts in all 6 comparisons (one restart per arm). That is text equality, not a bit-exactness
proof, and a 25k-token prompt read while the other stream decodes was not repeatable between two identical streams
of the same arm.
With a layer split, the engine options go into the config's `args`:

```
"args": [ ..., "--batch", "8", "--batch-groups", "4", "--trim-stage-weights" ],
"layer_split": "12,24,36"
```

| Option | What it does |
| --- | --- |
| `"parallel": N` / `--batch N` / `--slots N` | up to `min(N, kVerifyMaxT)` conversations have resident batch slots, with `kVerifyMaxT=8`; larger requests are held to that count with a diagnostic in every mode. More requests wait for a free slot. Each slot owns stage-local GDN recurrence, QSA K/V and indexer, and PLE history. Pipeline groups shape the admitted slots inside this cap. The ordinary floor is two slots; packed prefill permits one because its owner holds the window itself. Neither `--batch-mtp` nor `--batch-mtp-pipeline` raises the admission cap. |
| `--batch-groups G` | with a layer split (default: auto, below): the N slots in G groups that flow through the GPUs as a pipeline (GPU k runs one group while GPU k+1 runs another). G must divide N. 1 = all slots in one window, GPU after GPU. |
| `--batch-groups auto` | the layer-split default: the most groups, at most one per GPU stage, that divide the effective slots. `INFO batch_groups=G` reports the result. `--batch-groups 1` disables the pipeline. With `--batch-mtp`, multiple groups refuse unless `--batch-mtp-pipeline --batch-groups 2` selects the bounded MTP pipeline; `STRATA_BATCH_MTP_SERIAL=1` with no explicit groups chooses synchronous MTP. `--adapt-async 1` still resolves an unspecified group count to one. Historical ordinary-batch measurements, eight clients against one group: 4 x R9700 166 against 86 tok/s, 2 GPUs 109 against 78, 3 GPUs 110 against 78. |
| `--trim-stage-weights` | with an **explicit** `--layer-split` (e.g. `12,24,36`, not `auto`): every GPU loads only the dense weights of its own layers instead of the whole model's (the same as `STRATA_STAGE_TRIM=1`, PR #639). The VRAM this frees goes to the expert cache. Useful without `--batch` too. |

The fork's admission gate prices the **normalized requested** count (`min(N, kVerifyMaxT)`)
before allocation in every resident mode. The price includes private main-model and MTP
state, pinned staging, histories/checkpoints and transfer peaks. It refuses unsupported
resident KV formats or an over-budget plan. Runtime allocation may still reduce that count,
retaining the single-session fallback when fewer than two ordinary slots fit (packed prefill
permits one); the MTP pipeline instead refuses unsupported post-fit counts.
The server uses `INFO batch_slots=N`; `GET /v1/status` reports this effective `concurrency.serving`.

Each arrival checks **prompt + the original requested output + eight cells of slack** before position uploads,
request-identity binding or prompt/decode submission. `BGEN`'s internal `GEN 1` is only its prefill leg: it is
not the request's output budget. Main-session demand is the full, preallocated arena footprint of the
startup-resolved working/slot role, summed over each stage's own layer range. A shorter arrival neither
resizes nor re-resolves that placement. `ADMIT` reports that fixed device footprint, not an extra allocation;
draft, pinned-host and workspace budgets remain separately charged at startup. Invalid or overflowing lengths
refuse before work. A refused `BGEN` emits
`REJECT bounded=1 ...` followed immediately by `BADM <slot> 0`, with no output or new request identity.
The server consumes that exact boundary and frees only the rejected reservation; a missing or mismatched
boundary still ends the engine rather than risk assigning its lines to another request.

The HTTP `admission.active` count is a reservation count, not an execution count. `admission.waiting` counts
the bounded HTTP queue; `admission.native_waiting` counts reserved requests waiting to submit their next
native leg; `admission.executing` counts submitted request legs through their drained completion. The latter
does not claim simultaneous GPU kernels or overlapping prefill and decode. Active slot metrics include the
stable frontend `request_id`, matching `X-Strata-Request-Id` across solo/resident transitions and slot reuse.

### Client lifecycle adapter

`Service` renders/tokenizes requests and owns authentication, model/template selection, output dialects,
socket backpressure and deadlines. Generation crosses one in-process `LifecycleAdapter`:
`submit`, `events`, `cancel`, identity-fenced `priority`, and terminal drain. It uses the existing
stdin/stdout frames, with the logical native tag plus `(request, incarnation)` rather than a reusable seat.
Submission is not replayed after disconnect or engine exit; an accepted native leg has at most one terminal.
The existing bounded nonfinite-before-output retry remains a separate, explicit frontend recovery rule.

Only resolved retained-prompt **and** native-admission v1 capabilities select `native-retained-v1`.
That transport registers before preparation and delegates admission, seats and runnable policy to native code.
HTTP ingress remains bounded by effective slots plus queue depth, with separate bounded preparation permits
and the original request deadline. `LegacySeatTransport` is the explicit `legacy-seat` fallback for daily,
legacy and packed-only generation: its existing FIFO/seat policy, solo/image paths and fenced resident stops
remain in use. This extraction does **not** migrate daily traffic. Legacy live priority changes are refused
rather than sending an unfenced slot-only control; the per-request priority at admission is unchanged.
Score requests retain their separate bounded control lane.

Lifecycle/response attribution names the selected transport. Existing exclusive `frontend.seconds` and
`frontend.counts`, `native_phase`, `ingress_wait_s`, `native_admission_wait_s` and `native_service_wait_s`
remain the timing surfaces. Report preparation/cleanup and complete-call time separately from waits and
native-active time; preparation/cleanup include rendering and protocol work, so they are not an isolated
measurement of adapter dispatch cost. Mock-engine timings establish the partition, not GPU or installed-client speed.

The opt-in private packed/retained-prompt path now binds each reserved seat to a
`ResidentExecutor` call scope. `ResidentSlots` still owns identities, histories and model state;
the existing scheduler still selects rows and pumps controls. The shared final-row transaction
remains `run_slot_rows → commit_slots → wait_commit`, then the existing control pump, then
each selected owner's first-token/MTP catch-up. Every selected owner stays fenced until all
catch-up and residual/proposal consumers drain, including owners cancelled after issue.
Committed output is observed before the single terminal; an idle drain does not discard output.
Completed snapshot borrows are returned synchronously before seat replacement. Prepared and
stopped-owner captures keep their existing paths. `STRATA_EXECUTOR_TRACE=1` enables diagnostic
call/transaction receipts; it is off by default. This migration changes no daily profile,
admission/residency requirement or scheduling policy and makes no performance claim.

For a bounded synthetic acceptance cohort, `STRATA_BATCH_TRACE=1` writes complete native stdout frames as
`wire<` lines in the frontend log. Count only those trace lines' `T` and `BT` token IDs once; `ADMIT`, `BADM`,
`BDONE`, drafts and inert descriptors are not output. This opt-in trace does not change the default path,
and its arrival/promotion/refusal boundaries can be reconciled with HTTP usage and slot request IDs.

### Static expert residency

`--serve --batch` (including `--slots`, grouped MTP and resident multiplexing) assumes a **statically resident
expert working set**. The adaptive residency planner is inert in batch windows: the async tier is disabled
with batch slots, and the batch loop does not run the blocking planner. Neither `--adapt-swaps`, `--adapt-every`
nor `--adapt-async` can enable batch adaptation. Aggregate routing usage still accumulates; it does not change
the batch residency ranking. Misses therefore use the startup/first-use ordering for the whole batch run,
not a cache that learns from the concurrent requests.

After slot sessions, cache sizing/retries and all startup fills, the engine checks every `(layer, expert)`
pair in the **resolved model geometry** against its owning stage's cache and the peer/helper tiers. It counts
the union of actual pairs, not advertised slots, a sum that double-counts duplicate owners, or CUDA0 alone.
A truncated profile or a short stage cache cannot be hidden by spare slots on another stage. Dense stage
trimming does not reduce the model's expert set. If a profiled batch lacks any pair, or an unprofiled cache
cannot hold the full set, startup **refuses with `limiting=batch:expert_residency` and exit status 1**, before
`READY` or request admission. The refusal names static residency, the inert planner and the frozen miss ordering.
Use a complete `--expert-profile` and sufficient cache capacity across the owners, or omit `--batch`/`--slots`.
Fully resident batch profiles remain admitted without a new warning, including profiles with `--adapt-swaps 0`.

Two legitimate cases retain a loud warning rather than a new availability restriction: a full-capacity
unprofiled single-device cache still permits first-use filling, but does **not** claim established startup
residency; an explicit `VRAM` release under `--vram-elastic` is still honored and warns once if it leaves the
batch cache smaller than the expert set. Neither warning enables adaptive eviction. Outside batch serving,
partial caches are not refused: one startup warning reports the adaptation flags, since the solo/pipelined
planner is reachable there and still depends on those flags and its existing tier prerequisites.

`batch_residency_test` is a CUDA-free CPU test of these exact refusal/warning messages and the fully resident
24,576-pair regression, including split ownership, peer/helper unions and adaptation-flag independence.
It is metadata/declaration coverage, not a real-model run, performance measurement or stress qualification.


### Adaptive solo MTP versus ordinary batches

The default frontend policy is unchanged: alone, use `GEN` (solo MTP when loaded);
on an arrival, stop and continue through `BGEN`; when left alone, return to `GEN`
with at least 32 tokens left, at most twice per request. `STRATA_PARALLEL_SOLO=0`
still disables that return.

`STRATA_SPEC_POLICY=adaptive`, set in the **frontend's environment**, opts into a
measured policy over those same paths. It needs loaded `--mtp --spec >= 2`,
effective `--batch >= 2`, `--prompt-cache > 0` and a positive
`--foreground-budget-ms`. The policy refuses `--batch-mtp` and resident
multiplexing: its concurrent choice is the existing ordinary independent-request
batch, including split pipelines. Without loaded MTP it reports itself inactive
and retains ordinary generation; it adds no target-only command. Unset the
variable or use `STRATA_SPEC_POLICY=static` for the original policy.

Only received `T`/`BT` frames train the committed-output numerator. `DONE`,
`BADM`, `BDONE`, drafts and padding add no tokens. Context-length buckets and
effective load have separate decayed committed-token/wall-time estimates; eight
commits are required before an estimate can choose a path. Solo timing covers
`REUSED` through `DONE`, including proposal/verification/rejection/rollback and
terminal drain. Ordinary-batch timing includes intervening admissions, copies,
catch-up and zero-output wall time. These are process-local scheduling estimates,
not throughput evidence or additional reservations; restart discards them.

Unknown costs give waiting requests capacity immediately, as before. A measured
solo advantage above 10% may retain solo service for **at most one quarter of the
fixed foreground budget** and only while the existing control-queue deadline
has room for the observed handoff and a batch service interval. The burst's
deadline is fixed on first contention, not extended on each token. Due background
service and foreground HTTP demand not yet in the native queue force handoff.
When alone again, the predicted remaining-work saving must clear the 10% margin
and repay the measured first-token restore/catch-up cost before returning to MTP.
Stale solo costs are remeasured no more often than every eight seconds or eight
foreground budgets, whichever is longer; the existing per-request return cap
still applies. No performance gain or latency qualification is claimed before a
matched real-model run.

The policy changes neither sampler/acceptance nor state transfers, resolved
placement, admission ceilings, request identity or the native prefill partition.
The existing 2048-token cuts and tails remain fixed when that chunk is admitted.
It is not a new distributional guarantee: host acceptance tests establish their
own mathematical rule, not floating-point equivalence of all native dispatch
shapes. Sampled support and GPU parity retain their existing qualification bounds.

### Resident multiplexing without compute batching

`--serve --batch N --resident-multiplex` keeps the resident state allocation but
executes one request row per quantum on a single device. It does not advertise
batched arithmetic. `BPRIORITY <slot> foreground|background` selects weighted
round-robin service: foreground receives three consecutive quanta and background
one. With N continuously active slots, a background slot waits at most
`3 * (N - 1)` other decode quanta between turns. Prefill chunks remain separate
safe scheduling boundaries; this quantum bound is not a wall-clock latency claim.
No full conversation image is copied to alternate decode turns. Priority must be
set before each admission; the default is foreground. Multi-device pipelines are
not supported by this mode.

At a drained non-pipelined boundary, `BSTOP` completes cancellation without
executing another token for that row. Other resident requests continue. The
parent project's HET-036 evidence records the tested envelope; implementation
alone does not qualify HTTP concurrency, latency, or fairness.

### Explicit rows and idle resource diagnostics

The non-pipelined host path submits one descriptor per physical slot. Active rows
carry the bound request/generation, stable session pointer, token and position,
sequence extent, and output offset. Validation precedes execution; padding,
finished, and cancelled rows are compacted out and return `-1`, without sampling
or committing model state. Packed verifier residual indices remain distinct from
physical output offsets. This is independent-request batching, not speculative
positions from one request.

With the private `STRATA_BATCH_DIAGNOSTICS=1` gate, idle `BINERT` exercises all
three inert masks and compares resident fingerprints, history, positions and
sampling counters before acknowledging. `BRESOURCES release` drains and releases
batch graphs and batch-only device/pinned/history scratch; cached sessions and
solo/MTP resources remain. `BRESOURCES init` rebinds those stable sessions and
saved request contexts. Reinitializing without release is an error. The separate
`STATE_BATCH_STATE` diagnostic includes the actual committed verifier residual;
the existing `STATE_HASH` format is unchanged. Graph internals remain covered by
the admission allowance, not byte estimates inferred from graph counts.

`program::ResidentSlots` owns request identities, bounded histories and the
ordering of verifier, prefill, dispatch and private MTP transitions. The serving
loop selects work and emits protocol frames; it does not rebind those owners
independently. Admission and resource re-init require drained consumers. A failed
partial transfer leaves its target unavailable; a failed restore also prevents
continued use of the working session. Session and graph addresses remain stable.

A yielded request retains its identity and resolved automatic seed when its full
prompt and request settings are resent unchanged to the same destination, or when
the server promotes a solo `GEN` into the physical slot where `BYIELD` parked it.
A cancelled request or ordinary cached-prefix reuse receives a new monotonically
allocated identity. Released batch scratch still permits cached solo restoration;
resident admission waits for re-init, which never revives failed contexts. No
full-session copying is added to per-token scheduling.

`STATE_RETAINED` reports restored private bytes while batch resources are released;
its zero canary explicitly means no live batch-graph validation. `STATE_RESOURCES`
marks successful release/re-init boundaries, and re-init uploads and validates
retained owners again before live `STATE_SLOT` observations resume.

The protected lifecycle vehicle can explicitly set `STRATA_SESSION_ROWS_MAX=544`
for its expanded sequence; the default remains 256. Both envelopes retain 16 rows
per diagnostic file and the fixed routing/readback memory bound. This affects only
bounded diagnostic disk output, not the inference arithmetic or memory ceilings.


The project's protected B1 greedy cell exercised this path successfully. B2/B4,
sampling, configured EOS, and multiplex qualification are separate gates; this
result does not establish those capabilities or V100 runtime support.

### Request-owned expert groups

Expert execution already grouped a ready window's rows by expert, on both the
CPU pool and the primary GPU. It did not need another cross-request queue.
`ExpertRouteGroups` now supplies one CUDA-free scatter map to both paths:
each member retains its request generation, physical slot, logical position,
packed token index and original router index. Layer, operation generation,
immutable weight identity and gate/up/down formats belong to the whole window;
the final groups partition by expert and resolved execution worker. A helper
that selects only some rows produces separate worker groups, not a changed route.

Formation uses only rows already ready at the layer doorbell: **no formation
wait**, at most eight input rows and 128 routed entries. One fixed **2048-byte**
map is shared by all requests and stages, and charged once by admission; it does
not multiply the CPU pool, its scratch, or resident main/draft placements.
The existing single-resolved-placement pricing/carving contract is unchanged.
CPU jobs retain the eight-member cap, including repeated expert IDs.

Outputs remain **unweighted** at `(packed token * routed width + router rank)`.
The existing graph owns router coefficients and combines in the same router
order. No expert kernel, quantization, routing, shared-expert contribution,
sampling, speculative acceptance or reduction order changes. Group completion
order cannot become summation order. Exact affected-operator and same-prefix
real-model regression remain required; the CPU scatter oracle does not replace
them.

Cancellation keeps the existing single-host linearization: before submission a
cancelled row is omitted; an already launched window retains all member storage
and immutable weights until its consumers drain. Only then may a request retire
or its slot be reused. The frozen result identity is checked before host results
are exposed to the graph. There is no per-job cancellation that frees a member's
scratch while other members still use it. The asynchronous CPU oracle separately
exercises retiring one request after submission and discarding only its delivery,
without changing another request's speculative positions.

`STRATA_POOL_TRACE=1` reports actual routed entries, resolved groups, groups
spanning request identities, CPU jobs (`run`), and cumulative **submitted DMA
payload bytes** on host-serviced layers. These are separate quantities: shared
groups are not measured saved bytes or launch savings. Device-only all-resident
plans still bypass the host doorbell and are not counted by this trace. This
metadata change adds/removes no expert kernel launches and claims no speedup;
real-model overlap/traffic and a matched performance comparison need the GPU.
There is no new worker queue or dynamic worker-cost estimator.

The existing `expert_worker_test` includes a bit-exact grouped/separate
scatter/reduction oracle with mixed expert/worker sets, zero coefficients,
repeated IDs, permuted physical slots and multiple positions per request, plus
out-of-order asynchronous completion, partial retirement and stale
request/layer/span/format/residency refusal. `qsa_placement_test` and the parent's
native/Python planner fixtures continue to pin admission and resolved placement.


## Upstream grouped-slot records and the opt-in admission and decode switches

(From havok2-htwo's patch series, PR #2364.)

**Fork admission cap.** Every resident mode is bounded to `kVerifyMaxT` (8) admitted slots.
Upstream's grouped windows shape those admitted slots inside the cap; they do not admit
more slots in this fork. Packed prefill keeps its one-slot floor because its packed owner
holds the window itself. Neither MTP proposals nor the separate pipelined replay chains
change the admitted count.

Upstream's eight-row-window examples are `--batch 16` on four GPUs with four groups of four,
and `--batch 32` with four groups of eight (`--batch-groups auto`). Upstream reports that
each slot of its 16-slot batch matches solo tokens (`tools/batch_test.py --n 16`).
These examples and the measurements below are historical upstream records; their 16/32-slot
configurations are not admitted by this fork and are not qualification of its merged binary.

**Opt-in switches** (environment of the engine, i.e. `"env"` in the config; none is on by default). Each one changes
how the work is scheduled or copied, not what is computed; the identity gates and `tools/batch_test.py` are in
`phaseA-tests/review-havok.md`.

| Switch | What it does |
|---|---|
| `STRATA_SLOT_COPY_DIRECT=1` | an admission (and a slot's resume) copies the running state and the QSA K/V device to device, instead of through host images |
| `STRATA_SLOT_COPY_BATCHSYNC=1` | with the above: queue the ~116 copies and synchronize each device once |
| `STRATA_SLOT_CKPT_MOVE=1` | the admission's turn checkpoint moves into the slot (no 112 MB copy); it also keeps the dropped checkpoints' storage, as `STRATA_CKPT_POOL` does; a session SAVE or parked conversation taken right after an admission lacks the moved checkpoint (the slot holds it) |
| `STRATA_CKPT_POOL=1`, `STRATA_CKPT_THREADS=1` | a checkpoint is saved into the vectors of a dropped one; a split's stage parts are saved at once |
| `STRATA_ADMIT_MERGE_FIRST=1` | an admission reads its last prompt tokens and its first token in one window (all experts resident only) |
| `STRATA_PIPE_BYIELD_NODRAIN=1` | a BYIELD that reaches a pipelined batch is dropped without draining the groups |
| `STRATA_BATCH_SELF_COMMIT=1` | a one-row-per-slot window advances the GDN state itself, no commit graph |
| `STRATA_QSA_SLOTS=1` | a window's QSA block scores, top-k and attention run as one launch over its rows |
| `STRATA_HANDOFF_PIPE=P` | a layer split hands a prompt chunk to the next card in P pieces (2..16), gated by events, on a persistent thread |
| `STRATA_ADMIT_SHORT_FIRST=N` (server) | a new prompt of at most N tokens takes native admission control lines before longer ones within the existing foreground/background selector (a longer one that waited `STRATA_ADMIT_MAX_WAIT_S`, default 30, counts as short). Opt-in sorting cannot bypass background deadlines, bounded HTTP admission or physical-slot limits; absent it, existing FIFO order remains. |
| `STRATA_TRACE=1` | one line per admission with its phases (drain, prompt, first window, slot copy, checkpoint copy) |
| `STRATA_MMQ_DIRECT=1`, `STRATA_MMQ_GROUP_SORT=1` | CUDA, needs the build option `-DSTRATA_MMQ_TAB=ON`: the prompt MMQ reads resident experts in their cache slots; full MMQ groups sorted by row count (docs/DETAILS.md) |

Upstream measured (4x R9700 32 GB, HIP, IQ3_S all experts resident, layer split 12,25,37, 3 interleaved rounds per arm, 0.1.44.1
with 8 slots against the same build with 16 slots and the switches above; aggregate output tok/s, TTFT p50 in seconds):

| users (prompt) | 0.1.44.1, 8 slots | 16 slots, no switch | 16 slots, all switches |
|---|---|---|---|
| 8 (147 tokens) | 209 / 2.70 s | 209 / 2.72 s | 251 / 1.82 s |
| 16 (147 tokens) | 208 / 20.6 s | 252 / 5.4 s | 324 / 2.0 s |
| 16 (1.2K tokens) | 145 / 23.9 s | 167 / 10.6 s | 204 / 5.3 s |
| 16 agents x 3 turns, follow-up TTFT | 50 tok/s / 21.2 s | 53 / 7.1 s | 56 / 5.7 s |

One user alone is unchanged (73-74 tok/s). Each slot's session takes VRAM from the expert cache, so on a card or split
where the experts do not all fit, 16 slots cost the single-user speed (see docs above on what a slot costs).


### What a slot costs, and what setup recommends

Each resident slot owns a `SessionOwner` (GDN recurrence/convolution, QSA KV/indexer and PLE history), a private
MTP KV arena, sampling/steering settings, token history and bounded checkpoint. Mutable verifier, MTP and prefill
scratch is serialized under operation ownership; immutable weights remain shared. Admission prices the
resolved working-main, slot-main, working-draft and slot-draft placements, including fully resident
FP16/INT8/Q4_0/K8V4 pools and streamed slot intent. Unknown layouts and elastic K/V growth still refuse,
never count as zero. Requested payloads are charged before the automatic expert-cache grant.

The CUDA-free payload contract names four roles: `working_main`, `slot_main`, `working_draft` and
`slot_draft`, each with device and pinned bytes. Working owners are charged once; slot terms multiply the
normalized requested count (`min(N, kVerifyMaxT)`) in every resident mode.
Histories/checkpoints and serial transfer bounds remain separate, and shared drafter/head
storage is not charged twice. `plan/qsa_memory.hpp` shares format and pool arithmetic with the actual
allocator: a hybrid main state's host backing costs 3264 B/page, and its draft uses whole INT8 at 1056 B/cell
rather than hybrid's 816 B/cell. FP16 drafts remain FP16. These are source-derived payload prices pinned by
the project's hermetic planner/partner tests, not GPU allocation or concurrency qualification.

Streamed slots' full logical backing enters the existing guest/physical-host gate. With `--host-native`,
every process allocation is physical-host demand; VM mode counts unbacked guest allocation plus file-cache
mirrors without adding already-backed anonymous pages twice. Device, guest and host scopes, and startup
and runtime phases, each have to fit independently. A refusal reports `limiting=<scope>:<byte_class>`
(for example `host:resident_slot_main_pinned`) before any model allocation and before the runtime slot-fit
loop. Streamed slot allocation failure refuses rather than shrinking the requested set. Requested batches
also forbid an unpriced resident fallback when a working draft's pinned backing fails; solo retains its
existing logged fallback. Prefill remains bound to the working session, and solo promotion stays enabled:
the named `resident_transfer_transient` bounds the checkpoint and main/draft images in either direction.

`--slot-kv-window N` sets residency, in cells per attention layer, for the **additional** `--batch` slots.
Absent, slots inherit `--kv-resident` exactly as before. Explicit `0` makes those slots fully resident;
positive `N` keeps a page-rounded window in VRAM and the full logical K/V in pinned host RAM when the context
exceeds that window. The working main session and working draft retain the process-wide `--kv-resident`
policy. This is an explicit role override, not flag-order precedence; `--kv` and its rotation convention remain
process-global for main owners because the shared batch transforms use the working session's format flags.

Placement is resolved once at startup and passed to both role pricing and owner initialization. No live owner's
residency is rebound: captured row graphs retain its addresses and their keys omit residency. A slot draft
inherits the resolved working draft when the option is absent; batching cannot silently change that placement.
Explicit `0` makes it resident, and a positive override gives it a ring of that window. The window must hold the
existing `--mtp-window` plus `4 * max_t + 64` lookahead cells, capped by the context; an unwindowed draft must
hold the full context. Draft format selection is unchanged (whole INT8 under a hybrid process, not K8V4).
The four resolved role prices feed `working_main`, `slot_main`, `working_draft` and `slot_draft`; draft weights
and operation scratch remain shared.

The process-presence A/B overrides `STRATA_KV_MAIN_OFF` and `STRATA_KV_RING_OFF`
disable streaming for the working main/non-ring draft and working draft ring, respectively.
Even an empty or `0` value enables the corresponding override. Slots inherit these resolved
placements only when `--slot-kv-window` is absent; an explicit slot window bypasses them.
The Python planner and CUDA-free native partner apply the same rule when pricing these roles.

Startup refuses a negative or above-`INT32_MAX` value, a positive value below the **20480-cell engine floor**,
a window unable to hold one streamed call's `n_q * (cap / page_size + 2)` distinct blocks, or an undersized
draft ring. These are named option errors before device allocation, not the separate CUDA `RT = 1024`
clock-sweep guard or a later `ctl[3]` overflow. Explicit per-slot allocation failure also refuses rather than
silently shrinking the requested slots or switching their residency. Hermetic tests pin prices and bounded
carves; mixed-residency GPU execution/lifecycle qualification is separate from status reporting below.


The upstream setup heuristic recommends `"parallel"` where experts mostly fit in VRAM. It is a recommendation,
not an admission calculation or evidence for this fork. The historical measurements below show why slots can
reduce waiting while slowing a request alone: their private state reduces the expert cache. Use the emitted
admission plan for the actual context/slot count, not those historical per-slot memory estimates.

### Observing K/V placement

With `--slot-kv-window` set (including explicit `0`), startup emits additional `INFO` records for the working
session and each **effective** slot. `working_kv_mode` / `working_kv_window` and
`working_draft_kv_mode` / `working_draft_kv_window` describe the working owners; slot `N` uses
`slot_N_kv_mode` / `slot_N_kv_window` and `slot_N_draft_kv_mode` / `slot_N_draft_kv_window`.
Modes are `resident` (the whole K/V pool is in VRAM), `streamed` (a sparse main window or host-backed draft
ring), or `off` (no initialized state, normally no drafter). Windows are the actual page-rounded VRAM pool
capacity in cells per attention layer, including the full pool capacity for resident states; `off` has zero.

These records read initialized `SessionOwner::state()`, `MtpDrafter::kv_state()` and `slot_state()`, not the
option value or a predicted placement. A context that fits the window reports `resident`; a working draft
that used its existing resident fallback also reports `resident`. Without the per-slot option, the startup
protocol and frontend status fields are unchanged.

The frontend retains the native keys in JSON `/metrics.engine` and adds `kv_mode`, `kv_window`,
`draft_kv_mode` and `draft_kv_window` to each `/metrics.live.slots`, `/slots`, and `/v1/status.slots` entry.
`/v1/status.slots` is present only when the engine reports per-slot placement. The web About view labels the
working and each slot's main/draft placement separately, so a resident working session cannot make streamed
slots look resident. `STATE_RESIDENCY` is a request-retention diagnostic, not this K/V status source.

`kv_residency_info_test` checks the engine's CUDA-free serialization with supplied initialized-state fields.
`python3 -m unittest serve.test_server serve.test_parallel -v` exercises scripted INFO parsing and HTTP
reporting, including legacy field absence. These CPU checks do not prove native allocation, mixed-placement
execution, or GPU lifecycle correctness.

## Private resident multiplexing qualification

`"resident_multiplex": true` with an explicit `"parallel": N` adds `--resident-multiplex`. Unlike the batch
mode described below, this mode schedules **one resident row per quantum**: it is time multiplexing, not
simultaneous arithmetic. Requests use `BGEN` even when alone, without solo promotion/demotion. The native
scheduler, not the Python HTTP queue, enforces foreground/background quantum fairness. The service sends
`BPRIORITY` before every admission, including reused slots; HTTP admission separately uses bounded 3:1
foreground/background queues. See [DETAILS.md](DETAILS.md#bounded-admission-and-private-resident-multiplexing-het-036)
for request/config fields, private lifecycle/cancellation endpoints, budgets and protected smoke scenarios.
This is an opt-in qualification surface, not measured acceptance or a reason to enable public concurrency.
The multiplex service path explicitly refuses images; concurrent image-encoder work is not qualified.

Cancellation keeps the HTTP admission reservation through synchronous `STOP`/`BADM` or `BSTOP`/`BDONE`
draining. Slots and embeddings cannot be freed by a background timer while the engine may still consume
them. A missing boundary or ambiguous native admission refusal fails the process closed. Reused state is
resident; this service adds no per-token state swaps. Native metadata compaction, operation ownership and
safe-boundary retirement must be validated independently of HTTP correctness.
For a yielded, inactive request, the server instead sends `BSTOP` before releasing its frontend slot;
there is no `BDONE` for that parked state. This ends continuation ownership while retaining cached bytes.

## How the server uses the slots

- **One request alone** runs on the usual solo path (verify windows with MTP drafts): the fastest single stream.
- **When a second request arrives**, the first is stopped (`STOP`) and continues in a batch slot with its prompt
  plus what it generated so far - the engine's prompt cache holds exactly that, so nothing is read again - and the
  new request is admitted next to it. A request in a slot decodes **without MTP drafts** (one token per window).
- **A request left alone in a slot** (the others finished, nobody waits) goes back to the solo path: the slot is
  stopped, the engine copies its sessions back and decodes with MTP drafts again (at most twice per request; with
  `--prompt-cache 0` it stays in the slot; `STRATA_PARALLEL_SOLO=0` turns it off). Every resident commit advances
  that slot's private MTP KV using its own final residual and next token. Resume restores the same conversation's
  draft state; another conversation's draft KV is never an acceptable substitute.
- **More requests than slots** wait for a free one (`/metrics` -> `live.slots` shows each slot: idle, reading or
  decoding, its tokens and tok/s; `live.running` the requests in flight).
- **Each admission** reads the request's prompt through the usual prompt path (prompt cache and conversation
  checkpoints included) and produces its first token there; the state is then copied into the slot. Admissions
  are taken one at a time. The legacy policy decodes for half the previous chunk's duration
  (`STRATA_BATCH_DECODE_SHARE`, default 0.5); it bypasses interleaving/yield on pipelined groups and on an
  initial solo layer-split prefill. Use the time-budget policy below to make those paths schedulable.
- **A long prompt gives way to a short one** (#656's cooperative preemption): when a request with a prompt under
  half as long is waiting, the server sends `BYIELD`; at its next chunk boundary the long read stops, the part read
  so far is copied into a slot, the short request is admitted, and the long one then goes on from its slot with the
  same chunks (at most twice per request under the legacy policy; time-budget mode uses bounded service time).
- **Each slot is a private conversation cache.** After draining its consumers, a completed CUDA slot keeps its
  committed state and turn checkpoint only when prompt caching is enabled, the request permits checkpoints,
  its conversation scope is valid, and it has neither images nor failed state. The next turn must match that
  scope as well as exact tokens and steering mode. Scope affinity precedes pipeline-group hole filling; a new
  conversation uses the existing empty/LRU/group-spread policy. Request IDs, counters and sampling stay private.
- Slots are assigned so that consecutive requests land in different pipeline groups (`--batch-groups`).
- A client that disconnects stops its slot (`BSTOP`); the others go on.

### Measured time-budget scheduling

Opt in with `--foreground-budget-ms 1000 --background-share 0.25` in the engine arguments.
The positive millisecond value is a foreground scheduling target, **not** a promised first-token or
token-gap bound. `0` rolls back to the legacy policy. Share is finite and between `0.01` and `0.99`.
A positive budget requires serving, at least two admitted slots and prompt caching.
The existing bounded HTTP queue, context/output gates, `Retry-After` and queue-time metrics are unchanged.

The scheduler measures drained prefill quanta and complete decode rounds with the monotonic wall clock.
It selects how many **whole, memory-admitted arithmetic chunks** fit before the next decode deadline,
subtracting the measured decode cost. A cold estimate permits one chunk. A conservative decaying maximum
adapts to observed costs; it is an estimate, not worst-case execution time.
Changing the memory-admitted chunk invalidates its timing estimate: a cheap short-request chunk cannot
train a larger background chunk downward. The new geometry gets one cold indivisible chunk.
By default no workspace is enlarged, placement re-resolved, recurrent chunk resized, route changed or draft counted as output. Initial solo
reads and pipelined groups reach the same parking/interleave path. Each pipeline group runs once per
bounded decode round, then every stage drains before prompt/verifier scratch is reused.
The legacy `batch_read_piece` helper still selects one whole run for pipelined groups; only budget `0`
uses that selection to bypass the quantum loop. A positive budget instead selects quanta with `TimeBudget`.

`STRATA_PREFILL_RETAINED_LAYERS=stage` can separately opt into the
[retained-stage budget candidate](experiments/retained-prefill-model.md#opt-in-retained-stage-budget-candidate).
It keeps 2048-token chunks and the existing credit/debt decisions, with a sole
synchronous pump owner, bounded in-chunk decode and a full idle fence before
store or parking seams. It requires four fully resident slots, two MTP groups
and owned scratch on distinct devices; layer mode with a positive budget and
the optional 1024 split remain incompatible. This combination is not enabled
by default and has no GPU or performance qualification. It intentionally loses
the ordinary idle two-chunk fill, so cold/cohort throughput can regress.
Phase0 prefill receipts remain inclusive host spans; the scheduler's subtraction
of callback decode service must not be read as an additive diagnostic breakdown.

`STRATA_PREFILL_SPLIT_OVER_BUDGET=1024` separately opts into an execution split, not a smaller allocation.
It applies only with a positive foreground budget, an admitted 2048-token chunk whose measured cost
exceeds that budget, a currently served decoder, one selected contended chunk rather than pipeline fill,
and at least 2048 tokens remaining. That chunk runs as two 1024-token launches through the same loop:
controls, cancellation, store handoff, decode-first service and carried debt are revisited at the midpoint.
Together the halves reach the selected chunk's original endpoint; the original final tail is unchanged.
Midpoint parking keeps that endpoint with the request's resident slot. Only its exact full-image
continuation restores the pending second half; replacement, cancellation, failure and completion clear it.
Neither `Prefill::chunk()`, workspace, loans nor ring sizing changes. Each half earns background credit
from its actual elapsed time without training the 2048-token cost estimate downward.
Scheduling records for both halves include `split_over_budget=1024`; `indivisible_over_budget=1` still
means the **actual launch**, including a 1024-token half, exceeded the target.
Unset or `0` retains the original partition; other values are refused at startup, not used as a new
partition. No experimental subquantum below 1024 is selected (an unchanged final tail may be shorter).
While armed, the persistent arithmetic identity also binds the resolved split width, exact foreground
budget (including disabled budget `0`) and exact `--background-share`: earned credit can change whether
a decoder still contends at the next interval. Different split policies cannot restore each other's
recurrent images; with the split gate off, no new resolved identity fields are added.
This execution partition can change recurrent rounding and needs its own arithmetic/runtime qualification;
the CPU scheduling harness is not that qualification or a token-gap guarantee.

`ctest --test-dir <build> -R '^prefill_interleave_test$' --output-on-failure` checks this composed decision
on the CPU by compiling the actual prompt-read loop with synthetic GPU work and a deterministic clock.
It checks prefill/decode/prefill ordering, opt-in midpoint controls and cancellation, nonzero-frontier
continuation, exact prompt tiling, unchanged estimates and gate-off behavior, not device arithmetic or
pipeline-drain correctness.

`STRATA_TEST_PREFILL_SPLIT_OWNER=1` selects a dedicated block in the existing `strata-resident-test`
real-owner vehicle (single-device or three-stage host-native startup, admitted context at least 4096).
It parks an actual 1024-token resident prefix with pending endpoint 2048, restores the exact continuing
owner, and checks endpoint clearing on cancellation, replacement and one injected real-copy restore
refusal. It performs no token arithmetic; unlike the CPU scheduler's parking sink, these checks traverse
production `ResidentSlots::park` and `ResidentSlots::resume`. Removing the production parking assignment
must fail the nonzero-endpoint assertion. This fixture is not prefill/decode performance evidence.

Foreground decode credit comes from measured prefill time and is capped at one foreground budget.
An indivisible decode-round overrun remains debt for later quanta. Background control entries get a
deadline `now + measured_service * (1 - share) / share`, including chunk/park cost:
due background work precedes newer foreground arrivals; foreground entries otherwise remain FIFO
unless the explicit `STRATA_ADMIT_SHORT_FIRST` policy sorts new native admissions within that selector.
Foreground entries remain in the control queue until a physical slot is available. The bounded HTTP
queue also publishes a read-only pending-foreground scalar, including admitted but unsubmitted
requests: background cannot appear alone between decoder retirement and HTTP admission. That same
signal requests a park even when the foreground request has not reached native control yet.
Early background progress is work-conserving only when neither existing queue has foreground demand;
due background still advances under persistent arrivals. HTTP publishes under its existing status lock;
native control reads the scalar without acquiring that lock. No queue or dispatcher is added.
Repeated parking does not hit the legacy two-yield cap. Each resumed read must advance before it can
park again, so sustained arrivals cannot repeatedly restart the same prefix.
The API's cached-input count is fixed from the first native read, including a zero-output parked leg.
Later internal prefix reuse does not convert this request's own fresh prefill into cached-input credit.
Silent control reads also expose cooperative-yield polling every 0.5 s, rather than waiting for the
legacy ten-second HTTP heartbeat or for a completed `PP` chunk before sending `BYIELD`.

Without the explicit split opt-in, an indivisible chunk may exceed the target; the engine reports
`indivisible_over_budget=1`, never silently shrinks it or advertises success. Qualify any smaller execution
partition separately: changing it changes recurrent rounding. `STRATA_PREFILL_EQUAL`,
`STRATA_SPLIT_SMALL_OWN` and `STRATA_PREFILL_HELP` are refused in time-budget mode pending separate
partition/resume qualification.
Images and the final short tail retain their existing non-preemptible behavior. Saved scopes, identities,
RNG ownership, sampled-speculation acceptance and completed-slot retention keep their existing owners.

For opt-in evidence, `STRATA_BATCH_TRACE=1` includes host `monotonic_ns` stamps and request-to-slot
ownership marks. Native `T`/`BT` lines are committed token receipts; they are not draft, device-completion
or SSE-chunk timestamps. `STRATA_STATE_HASH=1` additionally reports every split stage at admission and
resume; resume hashes the restored working state (not just the unchanged saved slot), including logical
GDN/PLE/KV, draft KV on its owning last stage, history and sampling identity/counter. These draining
diagnostics cost time and remain off by default.

The following two upstream scheduling alternatives and their measurements are
retained from v0.1.45. Both are opt-in and off by default. The reported results are not
measurements of this fork's merged binary. Their interaction with packed prefill is
not measured by this project; no composition with `--foreground-budget-ms` is claimed.

## Reading a prompt beside pipelined groups (experimental, opt-in)

On the legacy path (`--foreground-budget-ms 0`), with a layer split and `--batch-groups`
above 1 a prompt is normally read in one run: every decoding slot stops until it is
read, however long it is (#1906). One GPU instead lets slots decode between chunks
for `STRATA_BATCH_DECODE_SHARE` x the chunk's time.

`STRATA_PIPED_CONCURRENT=1` reads a prompt on a thread of its own while the slot groups keep decoding through the
pipeline, instead of stopping them: the prompt's chunks and the slots' windows run on the same GPUs at once, on their
own streams (the windows' at the device's highest priority on CUDA; `STRATA_PIPED_CONCURRENT_PRIO=0` keeps the default),
so the slots never wait for the read and the read is one run with no drain inside it.

- The prompt's K/V goes to the main sessions and the slots' to their own, so they do not touch each other's state. The
  n-gram (PLE) table both of them read is locked. Upstream drains its adaptive VRAM round
  before the read and starts none during it; this fork's batch tier is static throughout.
- The VRAM expert slots the prompt path borrows are lent before the read and refilled after it, with nothing in flight;
  in between the slots' windows compute those experts on the CPU.
- The slots' doorbells are served on a thread of their own during the read, which makes no CUDA call. A CUDA call can
  wait for the driver's lock while its holder waits for the GPU - a free, or the module the first kernel of a new shape
  loads - and a window on that GPU waits for its doorbell (issue #31's pattern: two such hangs were found and read from
  the engine's stall dumps while this was built). The MTP drafter's prompt buffers and the prompt path's pinned upload
  slots are sized before the read, so its thread allocates and frees nothing. Between windows the doorbell thread waits
  for the next launch, which wakes it (a timed sleep is a whole timer tick on Windows, up to 15.6 ms). A window that
  will not finish is still found (its graph ended with a layer still to ring, or no layer rang for 20 s), and the stall
  watchdog stays on for the whole read.
- A waiting shorter request takes over at the read's next chunk (`BYIELD`), also when no slot decodes: the run ends
  there, the pipeline is drained and the part read waits in a slot, as off the pipeline. A read nobody waits for is not cut.
- Off, said once, where the read's thread would allocate or free (`STRATA_PREFILL_HELP=1`, `STRATA_SPLIT_MTP_BATCH=1`),
  where serving a doorbell makes a CUDA call (`--pcie-mode dma`, `STRATA_FS_SLOTS`, the helper caches
  `--expert-cache-device1..3`), and with an expert source that hands out buffers it recycles (the GGUF in place, an
  unbuffered source, `STRATA_IO_PREFETCH`): the experts come from the RAM arena or the mapped file. A prompt with
  pictures is read the usual way. This fork also keeps it off beside the foreground budget, batched MTP
  (`--batch-mtp`, `--batch-mtp-pipeline`), packed or retained prefill, `--resident-multiplex`, the judge lane,
  `--adapt-async` and `--spec-split`: each of those owns the slots or the drafter the reader would share, and the
  reason is printed once. One exception: a **retired** windowed ring (`--retire-resident-source`, whose fill is
  complete and which can hand out no more buffers) is allowed, because nothing reads from it during serving - the
  fully-resident case this mode is for.

One log line per read (`strata batch: read N tokens on a thread of its own in X ms (concurrent, to Q of T); the slots
decoded R rows meanwhile (S rows/s; stage k: W windows, H ms host each)`). Built for CUDA and HIP (HIP keeps the default
stream priority, and is not run on a GPU here); SYCL does not have it.

Measured on 2x RTX 5090 (Swift 1.5 IQ3_S, split 25, `--batch 6 --batch-groups 2`, resident experts, k8v4, 262K,
`--kv-resident 32768`, the engine-default decode share): main (`a0116b27`) against this mode, a fresh engine per arm in
the order main, this mode, this mode, main (`tools/piped_read_join_bench.py`; the fixed-prompt rows send the same four
stream prompts every run, so the streams' answers and work are the same):

| | main | `STRATA_PIPED_CONCURRENT=1` |
| --- | ---: | ---: |
| 64K + 32K cold prompts beside four streams: longest stop of a stream | 11.9 s | 0.77-0.84 s |
| ... the streams' tokens meanwhile | 289-291 | 1,411-1,679 |
| ... the prompts' TTFT | 6.8 / 5.8-5.9 s | 8.1-8.2 / 8.2-8.3 s |
| the same with fixed stream prompts: the streams' rate while the prompts are read | 2-3 tok/s | 85-92 tok/s |
| ... the streams finish at | 26.1-27.1 s | 24.6-25.3 s |
| ... GPU utilization while the prompts are read | 75-86% | 85-91% |
| a one-line question 1 s into a cold 45K read, nothing decoding: its TTFT | 5.7-5.9 s | 1.9 s |
| ... and the 45K prompt's own TTFT | 6.2-6.4 s | 7.4-7.5 s |

The joining prompts pay for the GPU time the streams now get. Over a whole run with the same work, the streams
finished 5-8% sooner (6.4% on average); with prompts that change from run to run (the bench's), the run's token rate
varies more than that between runs. Between windows the doorbell thread first slept 50 us, which is a whole timer
tick on Windows: the same build with `STRATA_TEST_BELL_SLEEP_US=50` gave the streams 64-67 tok/s during the reads
instead of 85-92.

The same randomised load (`soak`: three streams, cold joins of 4-64K, short questions, follow-ups of earlier joins,
clients leaving mid-read and mid-answer; 5 minutes, the same seed) on main and in this mode: 167 and 180 requests
answered; the longest gap inside a stream 9.8 and 1.0 s (90th percentile 8.5 and 0.7 s); the joins' TTFT median 3.8
and 4.6 s. A request that comes during a read still waits for it, unless its prompt is under half as long (`BYIELD`).

A 16 GB card on each side (`--vram-reserve-mib 16600 --kv-resident 16384`, the same flags otherwise):

| | main | this mode |
| --- | ---: | ---: |
| fixed stream prompts, 64K + 32K joining: the streams' rate while the prompts are read | 0.9 tok/s | 52.9 tok/s |
| ... the streams finish at | 70.6 s | 63.8 s |
| ... the prompts' TTFT | 9.5 / 10.1 s | 15.8 / 21.1 s |
| a one-line question 1 s into a cold 45K read: its TTFT | 7.3 s | 3.8 s |
| the 45K prompt read alone | 7.33 s | 7.25 s |

A weaker host as well (16 GB cards, the experts from the mapped file, AVX2 expert kernels, 6 pool workers), in #1906's
shape (`--batch 2 --batch-groups 2`): one stream decoding while another's cold prompt is read:

| cold prompt | main: the stream's longest stop / tok/s meanwhile / the prompt's TTFT | this mode |
| --- | ---: | ---: |
| 64K | 11.1 s / 2.4 / 10.9 s | 0.41 s / 25.7 / 11.7 s |
| 127K | 20.3 s / 1.5 / 19.9 s | 0.64 s / 32.7 / 20.7 s |

So the cost is the joining prompt's: +4-7% beside one stream, +20-45% beside four on these cards, and +67-109% beside
four on 16 GB cards, where the streams' windows take a larger share of the GPU.

**Checked:** with the variable unset this build gives main's tokens (six greedy answers' SHA-256 equal in two engine
starts each, `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0`) at main's speed. In this mode, 25 minutes of the
randomised load above (also on `--batch 2`): about 760 requests, 340 concurrent reads and 88 give-ways, with no hang, no
error and no stall report.

**Text equality** (`batch_interleave_test.py --long 40000` with its exact settings, `--batch 4 --batch-groups 2`, no
conversation cache): this test is not reproducible run to run on this PC even for main (A differed from its solo tokens
at token 142 in one run of two). In this mode every request equaled its solo tokens in two runs of three - D giving way
at its first chunk (`YIELDED 2 2048`), E admitted meanwhile, D going on and A going back to the solo path included,
which main does not do on this path (D never gives way there); the third run had B off at token 74, a near tie that
flips on main as well. Greedy only; other GPU pairs, three or more stages and sampled decoding not measured.

## A prompt read beside pipelined groups (opt-in)

On the legacy path (`--foreground-budget-ms 0`), with a layer split and `--batch-groups`
above 1 a prompt is normally read in one run: every decoding slot stops until it is read,
however long it is, `STRATA_BATCH_DECODE_SHARE` does not apply and no `BYIELD` is taken
(`batch_read_piece` returns 0 for the pipelined groups; #1906). One GPU instead lets
slots decode between chunks. These variables give the legacy pipelined path the same choice:

- `STRATA_PIPED_READ_PIECE=N`: a prompt read on the pipelined groups goes in pieces of N tokens, rounded down to whole
  chunks (at least one, at most 32), so a piece reads exactly the chunks a single run would, the two stages stay
  overlapped inside it and only the piece's end drains the pipeline. Between two pieces the slot groups decode through
  the pipeline for `STRATA_BATCH_DECODE_SHARE` x the piece's time, at most `STRATA_PIPED_DECODE_MAX_MS` and at least
  `STRATA_PIPED_DECODE_MIN_MS`, and every group in flight finishes before the next piece, as before any prompt read.
- At a piece's end the pipeline is drained, so a waiting request with a prompt under half as long can take over there
  (`BYIELD`, as between the chunks of a read off the pipeline): the part read waits in a slot and the read goes on from
  it after the short request is in. The pieces are also taken when no slot decodes, for this reason (off the pipeline
  the same read goes in `STRATA_BATCH_YIELD_CHUNKS` pieces then).
- A read that starts within a second (plus the decode budget) of the previous such read's end - prompts arriving back to
  back - first gives the slots one decode budget (the same rule, from the previous read's last full piece); otherwise
  the slots would stop for both reads.

With the variables unset the read is one run as before. One log line per piece (`strata batch: piped read N tokens in X
ms (to Q of T)`, with `, then the slots decoded R rows in Y ms` when slots decode), per give-way (`the prompt read gives
way at Q of T tokens; slot S holds it`) and per lead-in (`a read right after a read: the slots decoded R rows in Y ms
first`).

**What it trades.** Every piece ends by draining the pipeline, so a long read takes longer - also on an idle server,
since the pieces are what lets a short request in - and the decoding streams get tokens where they had none. It is a
latency choice, not a throughput gain: leave it off where throughput or the long prompt's own TTFT matters most, use it
where a stream that stops for several seconds, or a short question stuck behind a long read, is worse than a long prompt
that waits a little longer.

Measured on 2x RTX 5090 (Swift 1.5 IQ3_S, `--layer-split 25 --trim-stage-weights --batch 6 --batch-groups 2
--resident-experts`, k8v4 KV, 262K context, `--kv-resident 32768`, `STRATA_ARENA_PIN_GIB=24`, Windows 11; a CUDA 13.0
sm_120 build of this change on 0.1.43, "release" = 0.1.43 built the same way; a fresh engine for every run, all from one
window, `tools/piped_read_join_bench.py`). Four streams decode greedy coding answers (thinking off, up to 2048 tokens)
when cold prompts of this repository's own source text join. "Longest stop": the longest time any stream received no
token, from the first prompt's arrival to the last one's first token + 1 s (client-side, SSE); "stream tokens": what the
four streams received in that window; "token rate": the streams' tokens over the run's wall time.

Two cold prompts, 64K then 32K, arriving 6 s apart (`--scenarios two_joiners`; release and 32K pieces: median of 5 runs
each, interleaved; 16K pieces: one run):

| arm | prompts' TTFT | longest stop of a stream | stream tokens | token rate |
| --- | ---: | ---: | ---: | ---: |
| release (one run per prompt) | 6.9 s / 6.1 s | 12.2 s | 289 | 173 tok/s |
| `PIECE=32768 DECODE_MAX_MS=400` | 8.1 s / 8.5 s | 4.2 s | 527 | 166 tok/s |
| `PIECE=16384 DECODE_MAX_MS=400` | 10.5 s / 12.0 s | 2.6 s | 793 | 154 tok/s |

The longest stop was 12.16-12.23 s for the release in all five runs and 4.19-4.22 s for 32K pieces in four; in the fifth
the first 32K piece of the 64K prompt took 9.3 s instead of 3.95 s (one uninterrupted run before any slot decoded),
which made that run's stop 9.5 s. The token rate varied 169-180 tok/s (release) and 136-174 tok/s (32K) between runs.

A short request behind a long read, nothing decoding (#1504's test, `--scenarios followup,read_alone --repeat 3`, median
of 3): a one-line question sent 1 s after a cold 45K prompt, and the same 45K prompt read alone:

| arm | the question's TTFT | the 45K prompt's TTFT | the 45K prompt alone |
| --- | ---: | ---: | ---: |
| release | 5.4 s | 6.0 s | 5.7 s |
| `PIECE=32768 DECODE_MAX_MS=400` | 3.6 s | 7.3 s | 6.4 s |
| `PIECE=16384 DECODE_MAX_MS=400` | 1.8 s | 8.0 s | 7.1 s |

With the variables unset this change ran as the release in the same window (the 45K read alone 5.66 vs 5.69 s, the two
joiners' longest stop 12.2 s on both) and gave the same SHA-256 for six greedy answers under `--prompt-cache 0
--adapt-swaps 0 --pcie-frac 0`. For reference, one RTX 5090 (the same model and flags without the split, 6 slots) with
one cold 64K prompt joining four streams: `STRATA_BATCH_DECODE_SHARE=2.0` 2.4 s longest stop and 37.2 s TTFT, `=0` 14.0
s and 13.8 s. Pieces of one chunk (`PIECE=8192`) cost the long prompt the most. Not measured: other GPU pairs, three or
more stages, HIP and SYCL (built for CUDA only), sampled decoding, prompts with pictures.

## Historical upstream adaptive batch tier (not active in this fork)

This section preserves upstream's mechanism and measurements as historical context.
The merged fork instead requires the [static expert residency](#static-expert-residency)
described above: neither `--adapt-every`, `--adapt-swaps` nor `--adapt-async` enables
adaptive swaps in its batch windows. Do not use the following results to size or qualify
a partial-cache batch configuration on this fork.

Upstream's batch windows counted routed experts and, every `--adapt-every` windows
(default 4), swapped the most-routed missing experts for the least-routed resident ones,
using `--adapt-swaps` and `--adapt-decay` as in solo service. Each stage owned its swaps.
Rounds ran only for partial tiers with positive swap/interval settings, stopped during
prompt reads that could borrow cache slots, and did not run on pipelined `--batch-groups`.
Swaps changed GPU/CPU execution placement and could change greedy rounding.

With `--adapt-async 1` and the resident RAM mode, upstream advanced a round between windows:
the main thread changed tables and queued uploads, while a helper thread and refill streams
performed copies. An arrival drained the round before prefill. `--adapt-min-gain F`
(default 1.5) set the incoming/outgoing routing-count ratio; `adapt wait` recorded host wait
time. These are descriptions of the upstream implementation, not active fork batch flags.

Measured with the same setup as the `--batch-mtp` numbers above (2x RTX 3080 20 GB, UD-Q4_K_XL, `"layer_split": "23"`,
`"parallel": 2`, `--batch-mtp`, engine 0.1.41 with our production stack; two concurrent greedy decodes, the streams'
tok/s added, median of 20 measurements per restart):

| tier | aggregate tok/s | restarts | per window | routed entries served from VRAM |
| --- | --- | --- | --- | --- |
| `--adapt-async 1 --adapt-every 2` | **89.2** | 3 | 34.6 ms, adapt wait 0.2 ms | 92.1% |
| blocking, `--adapt-every 2` | 74.7 | 1 | 41.4 ms, adapt wait 8.4 ms | 92.3% |
| none, the cache frozen from a profile the tier had learned on the same prompts | 69.5 | 1 | 43.6 ms | 77.2% |
| none, `--adapt-every 0` | 45.2 | 2 | 66-68 ms | 52.7% |

The switch changes the tier in the solo path as well (there is no switch for batch windows alone), so the table
compares the tier as a whole, not "batch windows only": the solo decode speed moves the same way (81.7, 66.3, 69.0 and
43.1 tok/s). The prompts are the benchmark's own, eight topics that repeat, which the tier learns quickly, so on a
more varied workload the gain may be smaller (not measured). Upstream main runs no tier in batch windows, and this
was not measured against an upstream binary.

## Exactness

The historical upstream tests below established token equality for their own configurations; they are not this
fork's acceptance evidence. HET-035 qualifies two coexisting sessions on a single RTX 3090 using the pinned full
Flash-Next IQ3_S architecture. Its gate is exact committed full-vocabulary FP32 logits, actual routing IDs/weights,
tokens and logical state, including rejected-window restoration and cancellation/resume. The parent repository's
`evidence/HET-035/` records measured outcomes and limitations; a completed vehicle is not ticket acceptance.
No V100, multi-GPU, quantized/streamed resident KV, or arbitrary host-thread concurrency is implied.

The numerical comparison requires the same execution configuration on both sides:

- `STRATA_IQ_MT_MIN=1` (the multi-token CPU expert kernels for every group, as for the solo path's own
  exactness tests: by default an expert's rows round differently alone than in a group, so the output depends on
  how many rows of a window share an expert - which differs between a batch and a request alone),
- `--pcie-frac 0`: the PCIe share of the missed experts is chosen per window from the window's misses, so the
  same expert can run on the GPU in one window and on the CPU in another, which rounds differently, and
- `--adapt-every 1000000` (the VRAM tier fixed), and `--no-prefill-borrow` while a prompt is read beside decoding
  slots: their windows then see the expert cache without the slots the prompt borrowed (those experts run on the
  CPU).

With the default settings the outputs stay coherent but drift apart after some tokens, as two solo runs whose
windows differ can. Measured on the RTX 5070 (Q2_0, 4 conversations of 200 tokens, `tools/batch_test.py` without
the settings above): one equal to its solo run, the others apart from token 0, 99 and 174 - the first token already
differed for one, because the adaptive VRAM tier had moved experts between the solo runs and the batch; all four
texts read as well as their solo runs.

Sampled requests (temperature, top_p, top_k, min_p, seed) are drawn row by row with the solo window's
counter-based draw (Philox(seed, position)). Each resident row also stages its own bounded repetition/frequency/
presence-penalty history and steering flag. The shared process registry holds immutable steering tensors;
captured resident graphs consume explicit row flags rather than the solo request's global switch.

Resident initialization failure or slot-count reduction can retire the latest RoPE-table registration.
After all such setup/unwind paths, each device rebinds the longer-lived working session's immutable table
before graph capture. This preserves table-backed arithmetic (`STRATA_ROPE_TABLE=1`) in both the surviving
batch and the single-session fallback; resident teardown cannot invalidate a captured working-table pointer.

## Limits (for now)

- Batch windows carry no MTP drafts by default. `--batch-mtp` (or `STRATA_BATCH_MTP=1`) enables one proposal
  per resident request on one GPU or a layer split with each stage on its own GPU. At most four requests occupy
  the eight-row verifier quantum; a rotating cursor services larger slot sets. Split batch-MTP uses the existing
  synchronous verifier and accepted-prefix commit when groups resolve to one: every stage retains its scratch
  until target acceptance is known. It exposes `batch_schedule=synchronous`. Multiple groups refuse without
  `--batch-mtp-pipeline`; there is no silent override or feature drop. Ordinary batching keeps its group pipeline.
  Each request retains its private draft KV and uses the one serialized MTP scratch/head/weight owner.
  A proposal is kept only when the independently sampled target equals it; target Philox counters and per-row
  penalty histories remain request-local. This is not coupled draft sampling. Admission prices the eight-row
  verifier arena on every split stage; the synchronous schedule adds no retained rollback buffers.
  Separately, `--batch-mtp --batch-mtp-pipeline --batch-groups 2` retains **both** group pipelines on a
  distinct-device layer split with an even two through eight resident slots; other shapes refuse. Each group owns a verifier
  arena, row metadata, sampling/history controls and canary lease on every stage until the same recursive
  accepted-prefix commit completes. Four slots use four compute rows per owner and disjoint physical mapped
  handoff regions `[0,4)` and `[4,8)`, independent of active-slot compaction. Session and draft KV remain private
  per request, not duplicated per group. Engine status reports `batch_schedule=pipelined` and effective groups=2.
  Six/eight slots rotate through at most two selected slots per group; four compute rows and fixed handoff bases
  0/4 suffice. Selected slots are ordered canonically, bounding the four-slot group's pair/singleton depth shapes
  to 32 graph keys. Two/four-slot formation is unchanged. Compute capacity is `max(spec, min(batch, 4))`;
  larger resident counts increase private session/draft state and controls, not the retained replay quantum.
  A completed group waits for host-served expert graphs on other stages to drain before a synchronous prefix
  commit could strand their doorbells; fully resident stages retain continuous group overlap.
- Verifier graphs retain at most 32 row-order/base keys (window and commit executable per key). An unseen key
  beyond the bound refuses before capture; live graphs are not evicted.
- Repetition / frequency / presence penalties are not applied in batch windows.
- A prompt shorter than one chunk is read in one piece (the slots wait for it); a read gives way only at a chunk
  boundary, and not for pictures.
- Admissions are one at a time: two new long prompts are read one after the other.
- `--batch-groups` needs every stage on its own GPU. HET-042 retains completed CUDA pipelined slots: this fork's
  compacted group windows contain **only active rows**, and admission drains in-flight groups before mounting
  a prefix. Thus inactive sessions are not overwritten by pad rows. The old `finish(..., keep_cache=false)` was
  an obsolete restriction here. The separate SYCL pipeline still pads inactive rows and invalidates those
  caches; its non-retention is deliberate and has not been relaxed by this CUDA fix.
- The slot sessions take VRAM (above) and, with KV streaming, pinned RAM.
- Each requested role, including full streamed host backing, is charged before allocation.
  Unpriceable combinations and elastic K/V growth remain refused.

### Allocation and optional upstream modes

The owned prefill ring is one allocation. Every guarded owned buffer is charged in 2 MiB granules; raw token
and streaming identity allocations are independently rounded to that granule. Borrowed cache buffers retain
their 256-byte alignment/guards, and host staging/direct-read rings do not inherit device-page rounding.
Both prices use the same facts counter as the actual allocation sequence.

New allocation-bearing opt-ins remain off by default. Elastic KV growth, pipelined windows, asynchronous
adaptive refill, MTP Q4 conversion, and quantization-fusion workspace report unknown admission demand and
refuse until their allocation peaks are qualified. Their upstream implementation is retained; refusal is not
permission to silently fall back to a different requested mode. Batch-MTP prices its schedule-sized verifier
arenas through the same layout routine used by `Verifier::init`, in addition to private slot resources.
The pipelined mode additionally prices the second arena, resident controls and canary on every device,
per-stage mapped/history/diagnostic host staging and another finite graph-storage allowance inside the enforced
MPS ceiling. An unknown or over-budget plan refuses. Kernel exactness/performance still require parent-project
qualification; source integration is not runtime evidence.

### Split batch-MTP qualification, 2026-10-12

The parent project's `evidence/batch-mtp-split-20261012/summary.json` retains configurations, native token
sequences, admission prices, excluded arms and the slot-count curve. The adopted daily profile and its
protected frontend are unchanged; the new schedules are opt-in.

On engine `9c6d7b77`, four concurrent identical 224-token prompts with a 512-token cap, the same original
sender and frontend, and both `STRATA_BATCH_TRACE` and `STRATA_STATE_HASH` absent measured **137.1921 OFF
versus 156.9355 ON tok/s (+14.39%)**. The numerator is 2,048 API-committed tokens and the denominator is
last response minus first submission. Historical 131.19 used a different binary and is context, not this
pair's control. The separate 9,500-token/700-output cohort measured **82.4776 OFF versus 87.9263 ON
native committed tok/s (+6.61%)**, on the same binary/frontend/input within that pair. Its numerator is
`sum(BDONE.generated - 1)` and denominator the engine batch-window wall, including admissions. Native
per-request decode increased from 20.94/25.31/32.00/43.64 to 22.45/27.47/35.43/50.27 tok/s. These two
workloads and denominators must not be blended. The long cohort's newer frontend is measurement-only;
neither absolute figure qualifies the older daily frontend's scheduling/admission behavior.

The synchronous alternative was a measured negative on `903013ae`: ordinary pipelined batching
82.5555, synchronous without drafting 46.4209, synchronous with drafting 47.8359 native committed tok/s.
Drafting gained 3.05% against its synchronous control but remained 42.06% below the ordinary pipeline.
The previously traced timing figures were withdrawn; state-hash runs are correctness-only.

The combined greedy case offered 1,603 drafts and accepted 1,188: **415 rejected proposals** traversed
accepted-prefix commit across all three retained owners. All 2,800 emitted IDs matched the retained OFF
control. A diagnostic 32-token capture showed 30 rejections, including:
`request=3 slot=2 position=9500 proposal=6765 target=16107 keep=1`; emitted token index 1 was `16107`.
A separate non-greedy four-seed case (temperature 0.8, top-k 20, top-p 0.95 and request-local repetition,
frequency and presence penalties) matched all 160 control IDs and produced four distinct sequences.
This finite sampled parity plus source inspection of independent target Philox counters is evidence,
not a universal distributional proof; the concrete trace above is a greedy rejection case.

Admission charges an additional **67,440,368 B/device** for each second owner (67,234,560 B arena,
205,088 B controls, 720 B canary), another **256 MiB/device finite graph-storage allowance**, and
**303,174,720 B host-staging envelope**. The allowance/envelope are conservative prices, not measured
opaque graph allocations. Runtime arena allocation agrees with the ledger's 64.12 MiB. Private draft
KV was already allocated with the flag off, so it has no marginal cost at fixed slot count.
All 24,576 experts remained resident at N=4. A loaded capture observed free VRAM in CUDA0/1/2 order
of 3,861/4,853/2,428 MiB (physical NVIDIA indices 2/0/1). No reset or clock/power setting changed.

#### Production instrumentation boundary

The second-owner allocation, controls, graph allowance and replay lifetime are guarded by
`--batch-mtp-pipeline`; without it, no second verifier is initialized and ordinary `pump` is unchanged.
The off path retains pre-existing mandatory request/slot/epoch canaries, and small flag/null/deferred-mode
checks. The eight request-accounting records also reserve 192 bytes of host stack at serving setup;
they are production draft counters, not token/state capture. Do not describe the total off cost as zero.

`STRATA_STATE_HASH` gates hashes, proposal decision logs and canary/resource *printing*; it does not gate
mandatory safety checks. `STRATA_BATCH_DIAGNOSTICS` additionally enables diagnostic commands/dumps.
`STRATA_BATCH_TRACE` gates frontend token tracing. With these absent, there are no fingerprint copies,
per-token diagnostic output or diagnostic captures. Existing verifier profile/session-row/node/handshake
observations remain separately default-off; disabled observation retains only its pre-existing checks
and any fixed host bookkeeping. `GraphCaptureProbe` has a null thread-local observer by default: no
CUDA/RSS measurement or observation-forced upload/synchronization occurs. A nonempty graph's ordinary
node-count validation is pre-existing mandatory capture validation, not the optional node inventory.
The CPU resident preflight checks the absent peer, zero uninitialized-owner resources and disabled
capture observer under `CUDA_VISIBLE_DEVICES=-1`; it does not replace the runtime allocation ledger.

`STRATA_IQ_MT_MIN=1` is legitimate CPU expert arithmetic/performance policy (consistent rounding across
window widths), not instrumentation. `STRATA_PREFILL_RING=8` is the required bounded source-ring policy
for windowed-expert prefill, not instrumentation. Both remain in the daily environment.

#### Opt-in commit and admission host timing

Set `STRATA_COMMIT_TIMING=1` to append the existing per-stage, per-group verifier commit counters to
each pipelined batch-MTP completion report:

```
strata commit timing: stage=<n> group=<n> device=<n> scope=cumulative windows=<n> commit_ms=<ms> commit_ms_per_window=<ms>
```

`commit_ms` is the already-accumulated host commit term; the patch adds no commit clock or GPU
measurement. `windows` is that owner's existing completed-window count and is the exact denominator
of `commit_ms_per_window`. Owners with no windows produce no line. These counters are cumulative
over the verifier owner's lifetime, not confined to the cohort in the preceding wall-time report.
Group zero can also include earlier solo/admission windows; do not label its lifetime average as a
pipeline-only cohort average. Each stage's counter excludes the recursive next-stage commit.

Set `STRATA_ADMISSION_TIMING=1` for host-clock component lines alongside the existing `copied in`
transaction total:

```
strata admission timing: slot=<n> component=<name> device=<n> items=<n> host_ms=<ms>
```

The components are `checkpoint_select` (retained checkpoints examined), `validation_idle`,
`checkpoint_clone` (selected checkpoints cloned), `retire_checkpoint_bind`, `stage_sync`,
`checkpoint_transfer` (checkpoint save plus restore, once per stage), `main_kv_transfer` (all main
KV saves plus restores, once per stage), `draft_request_bind`, `draft_kv_transfer` (draft save plus
restore through the existing operation finish), `slot_metadata`, `initial_proposal`,
`context_canary_bind`, `publication`, and `total`. `device=-1` denotes host/control work.
`items` gives the checkpoint count, KV layer count, or prefix-cell count where relevant; otherwise
it is one operation (zero when no initial proposal was requested). Draft components are absent when
there is no draft KV. `bind` also serves parked prefixes, so their component lines do not have a
`BGEN` admission total. Only successful phases print; a failed transaction is not a complete sample.

All times use milliseconds with three decimal places. No extra CUDA synchronization, CUDA events,
GPU profiler, hashes, graph variant or persistent state is added. With both variables unset or `=0`, no
new clocks or lines run; the env gates retain only fixed host checks and stack timestamps.
The gates are cached once: unset or exactly `=0` is off; any other present value enables diagnostics.

For a diagnostic capture, set the gates in the engine launcher's environment and retain stderr,
for example `STRATA_COMMIT_TIMING=1 STRATA_ADMISSION_TIMING=1 <existing-engine-command> 2>host-timing.log`.
The engine's stdout protocol is unchanged and remains unbuffered. These instrumented arms are
**diagnostics only**: component printing is inside the existing admission transaction timer, and
observer/hash surfaces and their existing synchronizations can perturb its breakdown. The total
therefore includes logging, bookkeeping gaps and any separately enabled observers; it need not
equal the rounded component sum and is not a PCIe transfer-only price. Make no rate or timing
improvement claim from an instrumented arm against a clean baseline. Unset both gates, state hashing
and verbose token traces in every rate arm.

The draft-advance suffix is deliberately not timed. The current resident advance replays one
pre-captured graph: clocks around `record_front`/`record_rest` would measure graph construction,
not the removable replay suffix. Pricing that suffix requires a separate KV-only-suffix graph
behaviour variant and a clean two-build A/B with matching slot positions and inputs, not this
instrumentation patch.



## Historical upstream measurements (not fork qualification)

One RTX 5070 (12 GB), Ryzen 5 7600, 64 GB DDR5, Q2_0, 32K context, through the HTTP server: C different requests
sent at once (an 800-word essay each, 256 tokens per answer, greedy, thinking off), median of 3 rounds:

| Concurrent | Setting | Total tok/s | vs one at a time | Per request tok/s | First token: median / last of the round |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | one at a time (default; what setup recommends on this card) | 74.3 | - | 83.6 | 0.4 s / 0.4 s |
| 1 | `"parallel": 2` | 67.3 | -9 % | 74.8 | 0.4 s / 0.4 s |
| 1 | `"parallel": 4` | 57.9 | -22 % | 63.8 | 0.4 s / 0.4 s |
| 2 | one at a time | 71.6 | - | 77.3 | 2.1 s / 4.1 s |
| 2 | `"parallel": 2` | 61.0 | -15 % | 32.6 | 0.5 s / 0.9 s |
| 2 | `"parallel": 4` | 54.6 | -24 % | 28.7 | 0.6 s / 0.7 s |
| 4 | one at a time | 70.7 | - | 79.6 | 6.0 s / 11.2 s |
| 4 | `"parallel": 2` | 61.2 | -13 % | 32.2 | 4.5 s / 9.3 s |
| 4 | `"parallel": 4` | 63.1 | -11 % | 16.9 | 1.0 s / 1.8 s |

On this card the slots buy **waiting time, not speed**: the fourth of four requests starts after 1.8 s instead of
11.2 s, but together they decode 11-24 % slower than one after the other, and a request alone loses 11 % (2 slots)
or 24 % (4 slots), because the slots' sessions (0.56 GiB each) come out of the expert cache and most experts run
on the CPU: a batch window over 4 conversations reads 24 CPU experts per layer against ~8 for one, so it costs about
what the 4 tokens cost one after the other (`strata batch:` in the engine log: 54 ms per 4-row window, ~20 ms per
1-row window). This is why setup leaves a 12 GB card at one at a time. Cards that hold most experts in VRAM, and a
layer split, are where the slots also add speed (below).

A 4-GPU layer split (4 x 16 GB, PCIe Gen3), IQ3_S, `--batch 8 --batch-groups 4 --trim-stage-weights`, through
the HTTP server, 400 tokens per answer, temperature 0.7 (PR #559):

| Concurrent requests | Per request | Total |
| ---: | ---: | ---: |
| 1 | 123 tok/s (solo path) | 120 tok/s |
| 2 | 57 tok/s | 113 tok/s |
| 4 | 51 tok/s | 205 tok/s |
| 8 | 45 tok/s | 360 tok/s |

With the patches below on engine 0.1.38 and parking on, through the service: 8 requests at temperature 0 -> 369
tok/s, at 0.7 -> 358 tok/s.

`--trim-stage-weights` alone raised the share of experts held in VRAM on that machine from 76-85 % to 84-100 %
per card.

### One launch for all slots (opt-in)

In a batch window, QSA block scoring, top-k and decode attention ran as one launch per row, and the GDN recurrence
(in the window and in the 1-row commit) as one launch per slot. Two switches run each of them as one launch for all
rows, every row still reading its own slot's indexer keys, K/V pools and GDN state (passed by value as kernel
params, so the graph captured per row layout stays valid):

| Switch | What it does |
| --- | --- |
| `STRATA_BATCH_ATTN_ROWS=1` | block scores, top-k and decode attention of all the window's rows (up to 8) in one launch each. Not with KV streaming (a slot with `kv_mode` 1 keeps the per-row launches). |
| `STRATA_BATCH_GDN_GROUPS=1` | the GDN recurrence in the window, and the commit of 1-row groups, for all slots in one launch. The conv commits stay per slot. |

Both are off by default; off, the engine runs the same launches as before. On, the arithmetic per row is the same:
4 concurrent prompts at temperature 0 gave the same bytes as without them, with `--batch 4` and `--batch 8`.

One RTX PRO 6000 Blackwell (96 GB), IQ3_S, all 24,576 experts in VRAM, `--kv int8`, 262144 context, `--mtp`,
`--spec 4`, through the HTTP server: C different stories sent at once, 1024 tokens each, temperature 0.7, mean of
2 rounds, total tok/s (engine a0116b2, same day):

| Concurrent | `--batch 4` | with both switches | `--batch 8` (reserve 6144 MiB) | with both switches |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 179.2 | 179.4 | 186.6 | 179.7 |
| 2 | 204.1 | 218.7 | | |
| 3 | 243.2 | 276.6 | | |
| 4 | 268.4 | 311.7 | 267.7 | 311.6 |
| 6 | | | 297.7 | 354.8 |
| 8 | | | 308.4 | 374.8 |

The window (`strata batch:` in the engine log) costs 14.3 ms for 4 rows and 24.6 ms for 8 rows without them, and
12.2 ms and 19.9 ms with them: about 1.7-1.9 ms per added row instead of 2.4 ms.

## Together with conversation parking

`--conversation-cache-mib N --conversation-cache-slots K` (DETAILS.md) works with the layer split too: a request
whose conversation was parked is restored on every stage before its admission, so an agent and its sub-agents, or
several chats that alternate, come back without reading their history again. Measured on the same 4-GPU split,
two long conversations alternating through the HTTP server: the first turns took 3.9 s and 5.7 s to the first token
(their prompts read), the follow-ups 0.53 s and 0.46 s.

## Testing

The parent project controls authorization and resource bounds for GPU/model checks. Its current host-only
policy uses bounded normal-user runs with enforced memory ceilings; see `docs/03_BUILD_AND_BRINGUP.md`
and ADR 0002 in the parent repository. The older `tools/het035-session/USAGE.txt` protected vehicle and its
retained evidence describe historical qualification, not a requirement to restore VM-era infrastructure.

### Mixed-residency lifecycle fixtures

`conversation_snapshot_test` requires a CUDA device, but no model. It covers FP16, INT8, Q4_0 and
K8V4 in resident, sparse-streamed and ring fixtures, including genuinely provisioned hybrid host
K codes, K scales and V codes. It rejects each missing authoritative hybrid run before capture.
The ring fixture exercises the raw hybrid allocator/mover contract; the process's MTP draft format
under K8V4 remains whole INT8.

The whole-session cases restore resident → streamed and streamed → resident snapshots into independent
owners, compare every recurrent/indexer/main/draft payload, and compare authoritative fingerprints.
Streamed destinations begin with empty maps; a second restore invalidates a deliberately primed outgoing
map. Native allocator cases release and re-initialize the same arena for every format/residency pair,
checking secondary handles, pinned accounting, fresh owner identity and restoration after re-init.

`conversation_validation_test` and `conversation_transfer_test` need no GPU: their CUDA boundary wrappers
exercise invalid images and failed transfers using host storage. They are not model execution evidence.

The opt-in `STRATA_TEST_MIXED_RESIDENCY=1` block in `strata-resident-test` needs the full model, real MTP
weights, a CUDA device and admitted two-slot startup with `--kv k8v4 --kv-resident 0
--slot-kv-window 20480 --max-context 32768 --batch 2`. It temporarily binds the resident working session and
one streamed slot as two stable, independent `Verifier` row owners. Real graph capture and replay are
compared byte-for-byte against separate one-owner controls, including changed replay inputs, committed
residuals, picks and GPU request canaries. Padding/finished/cancelled peers and all-inert windows must
leave private state unchanged. Releasing/re-initializing batch graphs cannot free either owner's state.

That fixture also marks all four real INT8 draft code/scale runs, saves to a host-backed ring, overwrites
the resident working draft, and restores it without changing the peer draft. Its existing one-shot
`cudaMemcpy` fault is injected after a real GDN transfer but before PLE completes: the failed target stays
unavailable, the other owner stays byte-identical, and failed working restoration prevents further reuse.
Run with a bounded deadline and the parent project's enforced host/GPU ceilings. The legacy
`STRATA_SUPERVISED` test gate does not itself establish protected-supervisor evidence.

`kv_stream_parity` remains supporting **single-sequence** kernel evidence. It does not qualify mixed
row owners, independent requests, lifecycle transitions or concurrent HTTP serving.

### Protocol-driving checks

Four scripts drive a built engine or a running server; each exits non-zero on a failure. `serve/test_parallel.py`
tests the server's side with a scripted engine (no GPU).

| Script | What it checks |
| --- | --- |
| `tools/batch_test.py` | the same prompts alone (`GEN`) and together in the batch slots (`BGEN`): every slot's greedy tokens equal its solo tokens; prints the aggregate rate. `--batch-groups` in `--extra` tests the pipeline, `--keys "temperature=0.7"` the sampled rows. |
| `tools/batch_interleave_test.py` | a long prompt read while two slots decode, a prompt that gives way (`BYIELD`) and goes on, and a next turn continued from its slot: each equal to its solo tokens. On a two-stage layer split with pipelined groups add `--batch 4 --batch-groups 2` to `--extra` (its 3 slots do not divide into two groups) and set `STRATA_PIPED_READ_PIECE`; run it without a conversation cache, or step 4 restores the prompt instead of reading it. |
| `tools/parking_test.py` | a follow-up to a conversation decodes the same tokens whether its state stayed live or came back from the parking cache (with a layer split: every stage's image). |
| `tools/early_close_test.py` | a client that stops reading a streamed answer early (alone, and with a second request running) does not leave its tokens to the next request (server). |

For exact comparisons pass `--pcie-frac 0 --adapt-every 1000000` (and the scripts set `STRATA_IQ_MT_MIN=1`):

```
python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 8 --n 8 \
    --extra "--layer-split 12,24,36 --trim-stage-weights --batch-groups 4 --pcie-frac 0 --adapt-every 1000000"
python3 tools/batch_interleave_test.py --exe engine/strata --config strata-<model>.json \
    --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
python3 tools/parking_test.py --exe engine/strata --config strata-<model>.json \
    --extra "--layer-split 12,24,36 --conversation-cache-mib 8192 --conversation-cache-slots 4 --pcie-frac 0"
STRATA_KEY=<key> python3 tools/early_close_test.py http://127.0.0.1:8080
```

For single-GPU `--batch-mtp`, use a model with `rt/draft_vocab.bin` to exercise admission with a
shared draft-vocabulary head. This compares both slots' output against solo decoding:

```
python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 2 --n 2 --max-new 64 \
    --extra "--batch-mtp --pcie-frac 0 --adapt-every 1000000"
```

For CUDA split batch-MTP, `BDONE` also carries `draft_accepted` and `draft_offered`. Add
`--require-rejection` to the protocol-driving check to require a rejected proposal; retain and compare
the committed token sequences, not only its acceptance rate. `STRATA_STATE_HASH=1` additionally logs each
split proposal, independently sampled target, and committed prefix length for diagnostic runs.
Use the additional `--batch-mtp-pipeline --batch-groups 2` flags for the separately retained two-group path.
`STRATA_NAN_INJECT=<window>[:<row>][,...]` injects a nonfinite report into the numbered head window's local row
(row zero when omitted). Select each group's head window and a rejected proposal row when checking the
nonfinite contract: every verified row invalidates its whole slot, emits `BDONE nonfinite` instead of its tokens,
taints retained state and forces cold prefix reads. Deferred owners still commit/drain before slot retirement.
Run long greedy and independently seeded sampled requests against drafting-off controls and require rejection
in both modes. Keep state/token tracing in correctness captures only: unset **both** `STRATA_STATE_HASH` and
`STRATA_BATCH_TRACE` in every timing arm. Record the actual environment per arm and use native `BDONE`
request clocks or the engine's committed-window accounting, never shared frontend `last` statistics.

## Engine protocol (`--serve`)

On top of `GEN` / `GENI`:

| Line | Direction | Meaning |
| --- | --- | --- |
| `BGEN <slot> <max_new> [keys] <ids>` | in | read the prompt (as `GEN 1`), then continue in `<slot>` |
| `BPRIORITY <slot> foreground\|background` | in | set the resident slot's scheduling class before BGEN (multiplex mode); no success reply, malformed requests emit ERR |
| `BGENI <slot> <max_new> [keys] <file> <ids>` | in | the same with images |
| `BADM <slot> <1/0>` | out | after the admission's `DONE`: 1 = it continues in the slot, 0 = it ended; also the exact terminal boundary after a bounded admission refusal |
| `BT <slot> <id>` | out | a token of that slot |
| `BDONE <slot> <generated> <stop/length/cancel> <ms>` | out | the slot is free again; a non-pipelined slot may keep its conversation, a pipelined slot does not |
| `BSTOP <slot>` | in | retire an active slot at the next safe boundary with BDONE, without another token; for an inactive yielded slot, end continuation ownership without BDONE |
| `BYIELD <slot>` | in | the prompt being read gives way at its next chunk boundary; its part read waits in `<slot>` (the admission's own, or a free slot for a solo request) |
| `YIELDED <slot> <tokens>` | out | before the `DONE cancel` of a read that gave way: the request is sent again later and goes on from there |
| `INFO ... batch_slots=N` | out | the slots the engine runs (only with `--batch`) |
| `ADMIT cells=N state_demand_bytes=N context_limit=N batch_capacity=N slot_capacity_bytes=N admission_ns=N` | out | original prompt/output extent and summed stage-local main-session device arena demand/capacity, before work; timestamp uses the engine's steady clock |
| `REJECT bounded=1 <reason>` then `BADM <slot> 0` | out | a resident request refused before any output or identity binding; its slot remains reusable and other requests continue |

`tools/batch_test.py` drives the engine directly: the same prompts alone, then together, compared token by token,
and the aggregate rate.
