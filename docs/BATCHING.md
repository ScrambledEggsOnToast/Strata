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

With a layer split, the engine options go into the config's `args`:

```
"args": [ ..., "--batch", "8", "--batch-groups", "4", "--trim-stage-weights" ],
"layer_split": "12,24,36"
```

| Option | What it does |
| --- | --- |
| `"parallel": N` / `--batch N` / `--slots N` (2..8 normally) | up to N conversations have batch slots; more requests wait for a free slot. Each slot gets its own state (a session carved like the stage's own: GDN recurrence, QSA K/V and indexer, PLE history) on every GPU of the split. With grouped MTP, more than 8 slots can rotate through eight-row windows if memory permits. |
| `--batch-groups G` | with a layer split (default: auto, below): the N slots in G groups that flow through the GPUs as a pipeline (GPU k runs one group while GPU k+1 runs another). G must divide N. 1 = all slots in one window, GPU after GPU. |
| `--batch-groups auto` | the default on a layer split since 0.1.41 (give no `--batch-groups`): the engine pipelines one group per GPU stage (the most that divide the slots; 8 slots on 4 GPUs = 4 groups of 2) and says so (`INFO batch_groups=G`). `--batch-groups 1` turns it off (all slots in one window, GPU after GPU). Measured, 8 clients, total tok/s against one group: 4 x R9700 166 against 86, 2 GPUs 109 against 78, 3 GPUs 110 against 78. |
| `--trim-stage-weights` | with an **explicit** `--layer-split` (e.g. `12,24,36`, not `auto`): every GPU loads only the dense weights of its own layers instead of the whole model's (the same as `STRATA_STAGE_TRIM=1`, PR #639). The VRAM this frees goes to the expert cache. Useful without `--batch` too. |

The fork's admission gate prices the **requested** count before allocation, including private main-model and
MTP state, pinned staging, histories/checkpoints and transfer peaks. It refuses unsupported resident KV formats
or an over-budget plan. After admission, runtime allocation may still reduce the count, retaining the
single-session fallback when fewer than two slots fit. The server reads the effective count from
`INFO batch_slots=N`; `GET /v1/status` reports `concurrency.serving`.

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

### What a slot costs, and what setup recommends

Each resident slot owns a `SessionOwner` (GDN recurrence/convolution, QSA KV/indexer and PLE history), a private
MTP KV arena, sampling/steering settings, token history and bounded checkpoint. Mutable verifier, MTP and prefill
scratch is serialized under operation ownership; immutable weights remain shared. Admission prices the
resolved working-main, slot-main, working-draft and slot-draft placements, including fully resident
FP16/INT8/Q4_0/K8V4 pools and streamed slot intent. Unknown layouts and elastic K/V growth still refuse,
never count as zero. Requested payloads are charged before the automatic expert-cache grant.

The CUDA-free payload contract names four roles: `working_main`, `slot_main`, `working_draft` and
`slot_draft`, each with device and pinned bytes. Working owners are charged once; slot terms multiply the
requested count. Histories/checkpoints and serial transfer bounds remain separate, and shared drafter/head
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
  are taken one at a time, and **the slots decode between the prompt's chunks**: after each chunk (`--prefill`,
  2048-8192 tokens) they decode for half as long as the chunk took (`STRATA_BATCH_DECODE_SHARE`, default 0.5), so
  a long prompt slows the others down instead of stopping them. The chunks are the ones one uninterrupted read
  takes, so the prompt's arithmetic is unchanged.
- **A long prompt gives way to a short one** (#656's cooperative preemption): when a request with a prompt under
  half as long is waiting, the server sends `BYIELD`; at its next chunk boundary the long read stops, the part read
  so far is copied into a slot, the short request is admitted, and the long one then goes on from its slot with the
  same chunks (at most twice per request).
- **Each slot is a conversation cache.** A finished slot keeps what it holds (the prompt, the answer, and the
  checkpoint at the prompt's last turn boundary); the next turn of that conversation goes to that slot and the
  engine copies its state back (50-60 ms for a short conversation) instead of reading the history again - also for
  a client that drops the reply's thinking from the history (the checkpoint matches up to the new turn). A new
  conversation takes an empty slot, else the one used longest ago.
- Slots are assigned so that consecutive requests land in different pipeline groups (`--batch-groups`).
- A client that disconnects stops its slot (`BSTOP`); the others go on.

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
  per resident request on a single GPU. At most four requests occupy the eight-row verifier quantum; a rotating
  cursor services larger slot sets. Each request retains its private draft KV and uses the one serialized MTP
  scratch/head/weight owner. A proposal is kept only when the independently sampled target equals it; target
  Philox counters and per-row penalty histories remain request-local. This is not coupled draft sampling.
- Verifier graphs retain at most 32 row-order/base keys (window and commit executable per key). An unseen key
  beyond the bound refuses before capture; live graphs are not evicted.
- A prompt shorter than one chunk is read in one piece (the slots wait for it); a read gives way only at a chunk
  boundary, and not for pictures.
- Admissions are one at a time: two new long prompts are read one after the other.
- `--batch-groups` needs every stage on its own GPU. A finished pipelined slot is **not** kept as a conversation cache here: the fork's resident lifecycle clears it (`resident.finish(slot, keep_cache=false)`), where upstream 0.1.41 makes a pipelined slot a cache again — its #857 rule ships with the pad-row invalidation that this fork's compacted group window does not have. Carrying that rule is a batching-lane port item, not a re-pin change.
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
permission to silently fall back to a different requested mode. Batch-MTP prices its full eight-row verifier
arena through the same layout routine used by `Verifier::init`, in addition to private slot resources and
the existing opaque graph-storage allowance. Kernel exactness/performance still require parent-project
qualification; source integration is not runtime evidence.


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

## Engine protocol (`--serve`)

On top of `GEN` / `GENI`:

| Line | Direction | Meaning |
| --- | --- | --- |
| `BGEN <slot> <max_new> [keys] <ids>` | in | read the prompt (as `GEN 1`), then continue in `<slot>` |
| `BPRIORITY <slot> foreground\|background` | in | set the resident slot's scheduling class before BGEN (multiplex mode); no success reply, malformed requests emit ERR |
| `BGENI <slot> <max_new> [keys] <file> <ids>` | in | the same with images |
| `BADM <slot> <1/0>` | out | after the admission's `DONE`: 1 = it continues in the slot, 0 = it ended |
| `BT <slot> <id>` | out | a token of that slot |
| `BDONE <slot> <generated> <stop/length/cancel> <ms>` | out | the slot is free again (it keeps its conversation) |
| `BSTOP <slot>` | in | retire an active slot at the next safe boundary with BDONE, without another token; for an inactive yielded slot, end continuation ownership without BDONE |
| `BYIELD <slot>` | in | the prompt being read gives way at its next chunk boundary; its part read waits in `<slot>` (the admission's own, or a free slot for a solo request) |
| `YIELDED <slot> <tokens>` | out | before the `DONE cancel` of a read that gave way: the request is sent again later and goes on from there |
| `INFO ... batch_slots=N` | out | the slots the engine runs (only with `--batch`) |

`tools/batch_test.py` drives the engine directly: the same prompts alone, then together, compared token by token,
and the aggregate rate.
