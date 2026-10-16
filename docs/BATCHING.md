# Several requests at once (batch slots)

By default Strata serves **one request at a time**: the others wait in the server's queue. With `"parallel": N`
(the engine's `--batch N`, also spelled `--slots N`) the engine keeps up to N conversations open and decodes them
**together**: every verify window then carries one token of each conversation, so the dense weights, the shared
expert, the head and every routed expert two conversations share are read once per window for all of them.
Combined with a layer split across several GPUs and `--batch-groups`, the cards also work on different
conversations at the same time instead of waiting for each other.

It is opt-in and changes nothing when the options are absent (#465; the engine part is PR #559).

## Turning it on

One GPU: add `"parallel": 2` to the model's config (`strata-<model>.json`) and restart, or run setup with
`--parallel 2`. Setup recommends it only where it does not cost speed (below); any number you ask for is kept as
asked, with a note when it is more than setup would recommend.

```
"parallel": 2
```

With MTP (`--mtp` and `--spec`), `--batch-mtp` (in the config's `args`, or `STRATA_BATCH_MTP=1` in the server's
environment) lets each batch slot verify one MTP proposal per window. It is opt-in; without it the batch behaviour
described below is exactly the one without MTP. It needs VRAM per slot for the draft state and buffers, so check the
engine's free-memory log before using it on a smaller card. An incompatible explicit `--batch-mtp` request
**refuses startup**; it is not silently disabled. It needs at least two slots, `--serve`, loaded `--mtp --spec >= 2`,
and distinct devices for a layer split. The synchronous split route also refuses helper expert caches.
Historical one-GPU RTX PRO 5000 runs measured +31% to +39% total throughput with 2 to 4 clients (a RX R9700 run too).

**Native layer-split ownership:** one MTP scratch/head/weight owner on the last stage serves every slot;
each slot has private draft K/V, not a separate drafter object. A synchronous window verifies up to four slots
through the eight-row verifier quantum; six/eight slots rotate through successive windows. Select
`--batch-groups 1`, or `STRATA_BATCH_MTP_SERIAL=1` with no explicit groups, for synchronous split MTP.
The layer-split default otherwise resolves multiple pipelined groups when the slot count permits:
without `--batch-mtp-pipeline`, that combination refuses rather than dropping MTP or choosing serial execution.

`--batch-mtp --batch-mtp-pipeline --batch-groups 2` retains the fork's separate two-group route.
It admits **effective even slot counts 2, 4, 6 or 8**, on a distinct-device layer split, with no resident multiplexing.
Counts above eight normalize to eight **before** this even-count check: requested `--batch 9` can admit eight,
whereas an effective seven refuses. Runtime fit must still leave two supported groups. The pipeline's constrained
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
| `"parallel": N` / `--batch N` / `--slots N` | up to eight effective conversations have batch slots; larger requests normalize to eight, including grouped MTP. More requests wait for a free slot. Each slot gets its own state on every split GPU. Six/eight MTP slots rotate through bounded verifier windows. |
| `--batch-groups G` | with a layer split (default: auto, below): the N slots in G groups that flow through the GPUs as a pipeline (GPU k runs one group while GPU k+1 runs another). G must divide N. 1 = all slots in one window, GPU after GPU. |
| `--batch-groups auto` | the layer-split default: the most groups, at most one per GPU stage, that divide the effective slots. `INFO batch_groups=G` reports the result. `--batch-groups 1` disables the pipeline. With `--batch-mtp`, multiple groups refuse unless `--batch-mtp-pipeline --batch-groups 2` selects the bounded MTP pipeline; `STRATA_BATCH_MTP_SERIAL=1` with no explicit groups chooses synchronous MTP. `--adapt-async 1` still resolves an unspecified group count to one. Historical ordinary-batch measurements, eight clients against one group: 4 x R9700 166 against 86 tok/s, 2 GPUs 109 against 78, 3 GPUs 110 against 78. |
| `--trim-stage-weights` | with an **explicit** `--layer-split` (e.g. `12,24,36`, not `auto`): every GPU loads only the dense weights of its own layers instead of the whole model's (the same as `STRATA_STAGE_TRIM=1`, PR #639). The VRAM this frees goes to the expert cache. Useful without `--batch` too. |

The fork's admission gate prices the **normalized requested** count (`min(N, 8)`) before allocation, including
private main-model and MTP state, pinned staging, histories/checkpoints and transfer peaks. It refuses unsupported
resident KV formats or an over-budget plan. Runtime allocation may still reduce that count, retaining the
single-session fallback when fewer than two slots fit, except that the MTP pipeline refuses unsupported post-fit
counts. The server uses `INFO batch_slots=N`; `GET /v1/status` reports this effective `concurrency.serving`.

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


### What a slot costs, and what setup recommends

Each resident slot owns a `SessionOwner` (GDN recurrence/convolution, QSA KV/indexer and PLE history), a private
MTP KV arena, sampling/steering settings, token history and bounded checkpoint. Mutable verifier, MTP and prefill
scratch is serialized under operation ownership; immutable weights remain shared. Admission prices the
resolved working-main, slot-main, working-draft and slot-draft placements, including fully resident
FP16/INT8/Q4_0/K8V4 pools and streamed slot intent. Unknown layouts and elastic K/V growth still refuse,
never count as zero. Requested payloads are charged before the automatic expert-cache grant.

The CUDA-free payload contract names four roles: `working_main`, `slot_main`, `working_draft` and
`slot_draft`, each with device and pinned bytes. Working owners are charged once; slot terms multiply the
normalized requested count (at most eight). Histories/checkpoints and serial transfer bounds remain separate, and shared drafter/head
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
While armed, the persistent arithmetic identity also binds the resolved split width and exact foreground
budget, including budget `0` (disabled). Different split policies cannot restore each other's recurrent
images; with the split gate off, no new resolved identity fields are added.
This execution partition can change recurrent rounding and needs its own arithmetic/runtime qualification;
the CPU scheduling harness is not that qualification or a token-gap guarantee.

`ctest --test-dir <build> -R '^prefill_interleave_test$' --output-on-failure` checks this composed decision
on the CPU by compiling the actual prompt-read loop with synthetic GPU work and a deterministic clock.
It checks prefill/decode/prefill ordering, opt-in midpoint controls and cancellation, nonzero-frontier
continuation, exact prompt tiling, unchanged estimates and gate-off behavior, not device arithmetic or
pipeline-drain correctness.

Foreground decode credit comes from measured prefill time and is capped at one foreground budget.
An indivisible decode-round overrun remains debt for later quanta. Background control entries get a
deadline `now + measured_service * (1 - share) / share`, including chunk/park cost:
due background work precedes newer foreground arrivals; foreground entries otherwise remain FIFO.
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
## The VRAM expert tier keeps adapting in batch windows

Batch windows count which experts they route to, and every `--adapt-every` windows (default 4) the engine swaps the
most-routed experts that are not in VRAM into the cache in place of the least-routed resident ones, the same rule the
solo path uses (`--adapt-swaps` and `--adapt-decay` apply too; on a layer split every GPU's cache adapts). It runs when the
tier is not the whole model and both `--adapt-every` and `--adapt-swaps` are above 0. By default the round runs beside the
window's commit and drafts and has landed before the next window starts. It stands still while a long prompt is read
between decoding windows (the prompt may hold a loan of cache slots). The `strata batch:` log line shows the rounds and swaps, each stage's
GPU-reach wait and pool time, and how many routed entries the VRAM tier, the PCIe share and the CPU served per window.
Swaps move experts between the GPU and the CPU, which round differently, so greedy outputs can differ from run to run;
`--adapt-every 1000000` keeps the tier fixed (as in the exactness settings below). The pipelined `--batch-groups` path
does not adapt.

With `--adapt-async 1` (needs the resident RAM mode) the round does not hold a window up: each batch window moves the
round on one step (table changes and uploads on the main thread, the copies on a helper thread and the cards' refill
streams), as on the solo path. A request that arrives finishes the round in flight before it starts, and a prompt read
between decoding windows does not move it. `--adapt-min-gain F` (default 1.5) raises the bar a swap must clear (the
routing count of the expert coming in against the one going out), so fewer experts move per round; both tiers use it.
The log line's `adapt wait` is the time the main thread spent on the tier per window (the blocking tier: waiting for
its round).

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
| `tools/batch_interleave_test.py` | a long prompt read while two slots decode, a prompt that gives way (`BYIELD`) and goes on, and a next turn continued from its slot: each equal to its solo tokens. |
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
