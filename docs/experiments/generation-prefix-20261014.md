# Prefix generation candidates, 2026-10-14

The sections below retain the original worker's source-only delivery notes.
Subsequent coordinator qualification and its limits are recorded in the
[generation campaign](https://github.com/ScrambledEggsOnToast/ai-optimise/tree/main/evidence/generation-campaign-20261014).
Those independent results do not establish combined gains or a daily deployment.

Implementation and source reasoning only. No build, test, lint, formatter, GPU
probe, generation, service launch, download or performance measurement was run.
The coordinator owns all verification. Local commits deliberately bypassed hooks
because the dispatch prohibits tests and builds. Nothing was pushed or deployed.

Base: `335b954daa473fead32e17acd9e7e898944115f5`.

- `a7cbd65a`: direct selected-checkpoint restoration, independently applicable.
- `8fa97ab5`: immutable checkpoint ownership, separate opt-in treatment.
- `5a8d93b8`: selected draft-prefix assertions after ring wrap.
- `d65d4f1e`: distinguish live/cut recurrence and check failed ownership detach.

The last two commits add functional assertions, not executed evidence. The
ownership implementation does not require direct-cut activation; compare them
individually. No new translation units, CMake edits or external dependencies.

## Direct selected checkpoint

Set `STRATA_SNAPSHOT_DIRECT_CUT=1`; unset or `0` uses the original selection and
whole-image-then-rewind sequence. `STRATA_CKPT_REREAD` retains its original
separate diagnostic behavior and disables direct-cut activation.

The serving path validates the parent and all stage images, then resolves an
existing checkpoint at the selected token/image cut in every stage before
parking or writing any destination. The core restore API accepts only a pointer
to a checkpoint actually owned by that image. It validates the full image before
CUDA calls. This does not create a recurrent checkpoint at an arbitrary junction.

KV restoration validates the original captured extent, computes the selected
extent using the same format/layout code, and copies only those prefix bytes.
Packed pages still round upward; indexer pools retain their spare-row convention.
The selected checkpoint restores GDN, PLE and indexer state once, including
repairing the moving pooled row from its `dead` payload and reconstructing
`ple_prev`. Streaming maps reset and the draft ring refills at the selected cut.
The existing final drafter restore remains in place. Device addresses do not move.

The caller bypasses the redundant checkpoint rewind only after successful direct
restoration. It retains the original host KV extent for later validation/accounting,
limits unchanged reuse to the selected cut, truncates live identity to that cut,
and updates the selected checkpoint's LRU stamp. A transfer error remains fatal;
a partially restored owner cannot become reusable. Cancellation boundaries and
request ownership are unchanged; no asynchronous work survives these calls.

The existing restore log says `direct-checkpoint`, proving path exposure without
new per-layer instrumentation. `STRATA_SNAPSHOT_VERIFY=1` can compare the selected
draft extent/ring against the larger source image. This readback is a functional
observer, excluded from timings. Main-indexer readback against the original live
image's spare row is not a valid cut-state check: that row must instead match the
selected checkpoint's `dead` payload, as the host transfer assertion checks.

Retained capacities still grow/validate against the full parent image. This
candidate removes transfer work, not maximum-context reservations or host KV
allocation. Source selection and its cost prediction remain unchanged; use
`--prefix-selection longest` for the isolated comparison. Cost-mode observations
would change with the treatment and could subsequently change the selected source.

For live extent P and selected cut C, the removed transfer payload is
`KV(P)-KV(C) + one live running-state restoration`, summed over stages and draft,
with packed-page rounding and indexer-row correction included. Both paths still
restore the selected running state once. This is a source-derived work bound,
not elapsed-time evidence.

## Immutable checkpoint ownership

Set `STRATA_IMMUTABLE_CHECKPOINTS=1`, keeping direct-cut off for this comparison.
After synchronous capture completes, each checkpoint byte payload is sealed in
shared const storage. Checkpoint copies retain these allocations; IDs, image
identities, LRU, pins and stage directories remain independently owned. The park
helper shares already sealed payloads instead of copying them through the
watchdog-aware host-copy loop. Disk-loaded, unsealed payloads still copy on their
first park and seal the resulting copy; no disk-format change is made.

`CheckpointBytes::data()` exposes only const bytes. Builders use `write()`;
writing a sealed payload first makes a private copy. Allocation failure before
detach leaves both readers on the original allocation. Call sites retire builder
references before sealing; they must not retain a mutable reference across seal
or copy. The explicit mutable accessor is currently confined to capture, file
loading and focused assertions. A captured image is still published only after
its complete capture succeeds. Losing the source cache entry cannot destroy a
payload held by an incoming image, live chain or another reader.

Accounting is deliberately conservative: every reference is charged the entire
payload capacity plus its shared-owner representation. Sharing does not expand
the cache's eviction budget or evade the incoming-image reservation. This avoids
a second physical-allocation ledger and makes no capacity claim. Vector/allocator
bookkeeping remains subject to the existing accounting convention; sealed owner
representation is charged explicitly. The existing 4096-byte resident checkpoint
metadata envelope exceeds the source-derived object size on the targeted 64-bit
ABI; its compile-time assertion remains authoritative.

**Isolation limitation:** the wrapper adds a shared handle beside each byte
vector even when disabled. Thus the ownership build's metadata layout and
cache-admission byte counts are not identical to the base binary at exact budget
boundaries. Payload contents, deep-copy behavior when disabled and serialized
bytes are preserved by design, but default ABI/capacity equivalence is not claimed.
Use separate base/candidate binaries for that comparison, and additionally compare
off/on on the ownership binary to isolate sharing from the metadata representation.
No default rollout is requested. A live capture without retained checkpoint copies
can cost a few owner allocations without saving any copy; that negative case matters.

## Isolated comparison and finite assertions

Use the coordinator's frozen `generation-campaign-20261014/plan.json` and adjacent
`fixtures.json`: same model, rendered tokens, scope history, four maximum-context
slots, full expert residency, 2048 arithmetic partition, source selection, caps,
8 GiB reserve and thermal/resource stops. Fresh engine per arm, one matched short
warmup; no combined treatment in the initial screen.

| Arm | Direct cut | Immutable checkpoints | Purpose |
| --- | --- | --- | --- |
| Control | 0 | 0 | Whole restore, deep host copy |
| Cut | 1 | 0 | Only selected restoration changes |
| Ownership | 0 | 1 | Only sealed payload copying changes within this artifact |

The ownership base-binary comparison additionally exposes the metadata limitation
above. A combined `1/1` arm is an integration check only after independent screens.

| Finite scenario | Assertion and existing executable |
| --- | --- |
| Captured extent 9, selected cut 5, distinct GDN/indexer payloads | `conversation_transfer_test`: selected recurrence/PLE position, repaired spare row, untouched suffix past page 8, fewer copied bytes than full restore plus rewind |
| Foreign checkpoint pointer and partial transfer failure | `conversation_transfer_test`: rejection before CUDA/writes; partial mutation reports `transfer_failed`, never a reusable owner |
| Source capture with retained checkpoint | `conversation_transfer_test`: same payload bytes in both ownership arms; same backing pointer only when enabled |
| Sealed parent and stage parts copied, then parent evicted | `conversation_cache_test`: pointer identity and lifetime survive; pin/LRU stay private; a write detaches |
| Allocation failure during detach | `conversation_split_failure_test`: both original bytes and backing pointers survive |
| Draft image 65 tokens, selected cuts 4 and 5 after ring wrap | `conversation_snapshot_test`: authoritative selected bytes and resident ring agree across existing formats/modes |
| Serialization and stage split/merge | Existing `conversation_file_test`, `session_save_reclaim_test`, `conversation_split_failure_test`: unchanged serialized payload, retained ordering and failure rollback |
| One history root/older branch after displacement from all four seats | Actual `direct-checkpoint` exposure; same source/cut and processed/reused counts; complete activation includes baseline's later rewind |
| Live hit and cold request | Direct-cut negative controls; no false path exposure, no extra discarded history |
| Cancel restored continuation, then successor in same slot; different tenant/group | Drain/cancellation and scope isolation unchanged; no successor state from cancelled or foreign branch |

CPU tests are bounded by the existing 120-second test limits. GPU fixture has a
120-second limit. Run each relevant assertion once, stop on the first failure;
these are not soak or sampled-distribution qualification. No token/logit equality
across execution configurations is an acceptance gate. Byte assertions apply to
these arithmetic-preserving state movements.

Requested exact CPU invocations after the coordinator builds targets, with
`GEN_PREFIX_BUILD` set to its actual build directory:

```sh
: "${GEN_PREFIX_BUILD:?coordinator build directory required}"
env CUDA_VISIBLE_DEVICES=-1 STRATA_SNAPSHOT_DIRECT_CUT=0 STRATA_IMMUTABLE_CHECKPOINTS=0 ctest --test-dir "$GEN_PREFIX_BUILD" --output-on-failure --timeout 120 -R '^(conversation_transfer_test|conversation_validation_test|conversation_cache_test|conversation_split_failure_test|conversation_file_test|session_save_reclaim_test)$'
env CUDA_VISIBLE_DEVICES=-1 STRATA_SNAPSHOT_DIRECT_CUT=0 STRATA_IMMUTABLE_CHECKPOINTS=1 ctest --test-dir "$GEN_PREFIX_BUILD" --output-on-failure --timeout 120 -R '^(conversation_transfer_test|conversation_validation_test|conversation_cache_test|conversation_split_failure_test|conversation_file_test|session_save_reclaim_test)$'
```

The direct core API assertions execute in both arms; the engine switch only
controls serving selection. Run the GPU fixture with the coordinator's normal
host monitor inside its exclusive `window.py run --class measure --wait-s 1800
--max-s 1800` window, never concurrently with a model or build. All future builds
use the shared `window.py` build class, including HIP/SYCL dependency closure.

Exact serving treatment is the profile `env` mapping in the arm table, with
`--prefix-selection longest` and all other frozen engine arguments unchanged.
The runtime binary and profile paths must be pinned by the coordinator after its
build; no executable artifact exists from this worker. **Harness limitation:** the
inspected ordinary `tools/host-inference/measure.py` cohorts explicitly refuse
`--prompt-cache > 0` except their responsiveness path, so `--cases tiny` or `smoke`
cannot establish these history assertions. The coordinator must use its monitored
history campaign driver; no invented cache CLI flag or unsupported complete runtime
command is supplied here. This is missing runtime evidence, not a failed candidate.

## Shared prefix extents: evaluated, not implemented

The existing `ConversationBuffer` owns 16 MiB segments uniquely. Retained KV moves
from one incoming snapshot into one session's reuse allocation. Equality of token
IDs is not ancestry: only restored storage bounded by `unchanged_tokens` establishes
unchanged bytes. `conversation_kv_save` preserves whole pages strictly before the
first dirty token, refreshes the partial page and indexer's moving spare row, and
refreshes the draft's last possible unwritten cell through `unchanged-1`.

A const reference to that existing segment cannot safely be shared with a sibling
while the existing mutable `visit`/`resize` interface writes it. Full safe extents
would require explicit sealing, copy-on-write for every segment intersecting the
dirty page/spare row, and physical allocation accounting through retained, incoming
and in-flight owners. Large segment alignment can duplicate substantially more
than a four-token tail. Changing only the segment pointer type falsifies the
"free sharing" premise: it creates shared writable bytes. This does not falsify
an immutable extent design.

For m siblings sharing an actually retained root r, a storage upper opportunity is
`(m-1)*KV(r)`, plus root checkpoint copies already addressed separately here. Host
sharing removes no H2D traffic into the unchanged private GPU arenas. At m=1 there
is zero duplicate-root saving; the existing retained-buffer optimization already
omits unchanged recapture. New gain cannot be credited for those same bytes again.
Actual duplicated ancestry/fanout and physical savings are unmeasured. This remains
a conditional larger candidate, with no claim of rejection on performance grounds.

## Stage-ordered async transfer: evaluated, not implemented

`copy_nonblocking` submits to a private nonblocking stream and immediately joins
it. `conversation_kv_restore` then resets streaming maps/refills draft rings on the
null stream. `conversation_checkpoint_restore` has device joins and repairs the
indexer's spare row. The serving caller completes every stage before entering the
prompt reader, and destroys incoming snapshot metadata at that synchronous boundary.

Removing only stream joins would permit ring/reset/restore consumers to race DMA
and temporary snapshot storage to expire before DMA completion. A private stream
alone creates no ready-stage publication contract. Merely submitting faster while
keeping the all-stage read barrier can reduce submission overhead but cannot overlap
earlier-stage prompt compute with later-stage transfer: that compute is unreachable
until the barrier. These narrow premises are rejected by source ordering.

For a complete redesign, with readiness times Xj and compute service Pj, the
ideal stage chain is `Fj=max(F(j-1),Xj)+Pj`. If the all-stage barrier remains, its
compute start is at least `max(Xj)`, so stage-readiness overlap is zero. Without
the barrier, gain is bounded by transfer waiting removed from that recurrence;
unchanged bytes and link contention remain. No durations are assumed or simulated.

A safe implementation needs per-stage outgoing/incoming leases through completion,
whole-plan validation before writes, admitted bounded staging, consumer event waits,
last-stage draft completion, and cancellation draining all issued tickets before
reuse. No writable-session alias or additional live model replica is acceptable.
This worker does not introduce that scheduler/lifetime contract; its safe end-to-end
speed premise is unmeasured, not disproved. The two implemented candidates remove
work while preserving existing synchronous lifetimes.

## Coordinator path witness

`STRATA_CHECKPOINT_COPY_TRACE=1` reports the actual five payload pointer aliases
after each `copy_checkpoint_beating` operation as `CHECKPOINT_COPY`, with shared
and copied byte counts. Child stage parts report their own payloads; the parent
does not count those bytes again. Token/image metadata, other copy sites and GPU
transfers are outside this witness. Use it only for functional path exposure;
leave it unset for timing. Sharing retains the source allocation rather than
claiming that those bytes disappeared from all owners.

## Build closure and remaining risks

Changed implementation files: `include/strata/core/checkpoint_bytes.hpp`,
`include/strata/core/conversation_cache.hpp`,
`include/strata/core/conversation_snapshot.hpp`, `src/core/conversation_copy.hpp`,
`src/core/conversation_file.cpp`, `src/core/conversation_snapshot.cpp`,
`src/core/conversation_state.cpp`, `src/program/generate.cpp`.
Changed assertion files: `src/core/conversation_cache_test.cpp`,
`src/core/conversation_snapshot_test.cpp`, `src/core/conversation_split_failure_test.cpp`,
`src/core/conversation_validation_test.cpp`. This document is the only report file.

Static quoted-include closure finds these production translation units:
`conversation_file.cpp`, `conversation_snapshot.cpp`, `conversation_state.cpp`,
`mtp.cpp`, `generate.cpp`, `resident_slots.cpp`, `serial_judge.cpp`.
Rebuild their libraries and relink the engine and dependent executables. Test
closure additionally includes all changed assertions plus `conversation_file_test`,
`session_save_reclaim_test`, `residency_model_test`, `resident_model_test`, and
`resident_slots_test`. Compiler depfiles remain authoritative for generated/backend
includes. The checkpoint layout changes ABI; an overlay must not mix old callers
with new headers. CUDA, HIP and SYCL builds remain unperformed, not passed.

Runtime multi-stage validation, emitted-path proof, cancellation under a real
client, real cache/accounting pressure, and timing all remain unverified. No source
inspection proves a speedup. Conservative sharing charges may cause earlier eviction
at exact byte caps, and unsealed disk checkpoints still incur their initial copy.
The full parent-capacity validation intentionally remains in direct-cut mode. Keep
all safety stops and do not retry an unchanged failure. Follow the coordinator's
frozen usefulness/stopping rules; no daily/default promotion follows this delivery.

## Optional integration

The integration preserves separate, disabled-by-default controls:
`STRATA_SNAPSHOT_DIRECT_CUT=1`, `STRATA_IMMUTABLE_CHECKPOINTS=1`,
`STRATA_PREFIX_BACKING=1`, `STRATA_SHARED_ROPE=1`,
`STRATA_PREFILL_RETAINED_LAYERS=1`, `STRATA_RESIDENT_DIRECT_COPY=1`,
`--batch-mtp-floating`, and `--batch-mtp-third`. The last two require their
existing batch-MTP pipeline prerequisites. Cost-aware HTTP dispatch remains
a separate frontend/host option, not an engine default.

Conflict resolution retains selected-cut byte extents, copy-on-write checkpoint
destinations, copy-observer classifications, and each unsupported SYCL-mode
refusal. The asynchronous stage-restoration experiment is not included: its
matched screen showed no useful gain. Eager context tiers are not included.
Shared backing is a retained-memory option, not a transfer-byte or speed claim.
Native handoff improves the exercised mixed cohort's completion; its first-token
and output-gap effects were not uniformly beneficial in both timing pairs.

Keep profile promotion separate from source integration. The campaign record
identifies the actual build, checks, combined exposure, exclusions and rollback;
this source document does not grant qualification to unexercised combinations,
hardware backends or long-context quality.

The integration review also requires SYCL checkpoint writers to request mutable
payloads and SYCL restore definitions to match the shared selected-cut API.
Source compatibility repairs do not establish a successful SYCL build. Combined
qualification must cover shared-prefix donor isolation during selected restore
and repark, and retained-prefill cancellation with the third owner active;
individual feature results do not establish those interactions.
