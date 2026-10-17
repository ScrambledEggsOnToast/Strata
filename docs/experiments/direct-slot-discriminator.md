# Resident handoff: direct-copy candidate

The original source-only candidate used base
`498d28b7b9ea2bfd27afd0e24b4eb718f5291353`. Its supplied observer baseline
reported 783273504 successful transfer bytes without copy failures; that was
not a measurement of this candidate. The 2026-10-17 selected-cut verification
below now exercises the corrected legacy and direct paths on real three-GPU
owners. It establishes bounded movement/lifecycle correctness, not a speedup
or daily/default promotion.

## One switch and bounded scope

Set **`STRATA_RESIDENT_DIRECT_COPY=1`** in the isolated engine profile. Only the
exact value `1` enables the native CUDA/HIP path; unset, `0`, or any other value
uses the existing snapshot path. Selection is frozen once before native memory
admission and shared by every later main/MTP handoff, even if the environment is
subsequently changed. SYCL explicitly refuses `1` at startup. Keep all existing
capacity, arithmetic, residency and scheduling flags unchanged.

This is a copy optimization, **not zero-copy binding or direct-to-slot prefill**.
The working prefill/first-token verifier and resident slot graphs keep their
existing owners and stationary addresses. Main admission and live resume copy
running state and authoritative KV directly between those owners on each stage's
existing device. Draft save/resume uses the same direct KV helper. There are no
additional model replicas, graphs, reservations or rebinding operations.

The helper reuses snapshot layout/validation, format identity (including rotated
INT8 and hybrid K8V4), authoritative-pool selection, the existing nonblocking copy
stream, and stream-map/ring repair. KV calls retain the existing 16 MiB segment
limit and progress notifications, without allocating snapshot segments. A
resident-to-resident payload is a same-device copy. Streamed/ring owners still
use their **existing authoritative mapped host pools**; that traffic is not
claimed to be D2D. No path stages an intermediate CPU image or falls back to one
on mismatch/failure. True host-park restoration and selected checkpoint storage
remain host-backed.

Old-consumer joins, request binding, cancellation, publication and failure
invalidation remain with `ResidentSlots` and `MtpDrafter::Operation`. A failed
main admission invalidates its destination; a failed resume also poisons working
reuse. Failed draft operations retain their existing poisoned scratch ownership.
Map/ring repair in the new helper is synchronized before publication, including
MTP's nonblocking consumer stream. Both copy paths resume a selected checkpoint's
KV first, then its saved GDN/PLE/index tail and spare-row repair: copying the
longer-lived slot's pooled rows last would overwrite the selected cut's repaired
row. The legacy live-resume order and all existing fences are unchanged.

`resume_work` prices one transfer leg for direct state, two for the old bounce,
and retains the host checkpoint restore and spare-row charge. Its direct route
has a separate cost identity; no scheduling policy is changed.

## Selected-cut resident regression and verification

`resident_host_model_checks` now parks a real retained owner beyond the selected
index block, then resumes that owner's saved checkpoint through
`ResidentSlots::resume` / `copy_from`. It covers an aligned cut at `idx_block`
and a partial-block cut at `idx_block + 1`, with the retained prefix at
`2 * idx_block + 1`. Every owned QSA layer on all three stages has an explicitly
different completed pooled key and selected checkpoint `dead` key; the fixture
reads both on their owning devices and requires the selected key after resume.
Full main/draft snapshots, the unchanged retained source, identity/seed, peer
payload/identity, invalid-cut refusal before writes, and stationary addresses
remain checked. The existing terminal partial-copy poison checks still run.
This is synthetic lifecycle state on the real allocated owners, not an inference
arithmetic, throughput, or backend-qualification claim.

On 2026-10-17, the test-only discriminator on
`fcd3b1034e4d1da759f462eed0a65d61d9f28110` failed the legacy selected pooled-key
assertion: 258 checks, one failure. The corrected fixture subsequently passed
**572 checks in each of the legacy and direct arms**, with cuts 4 and 5,
advanced prefix 9, three stages and four seats. Each arm emitted both
`RESIDENT_CUT_REPAIR ... failures=0` records and completed owned process/MPS
cleanup without errors. The unfixed direct path was not run in this comparison.

Retained captures are under
`/home/joshkirklin/strata-workspace/scratch/bottleneck-trials-20261017/`:

- `resident-cut-red-r2-run/result.json`: original failing legacy discriminator.
- `resident-cut-green-r3-legacy-run/result.json`: corrected legacy path.
- `resident-cut-green-r3-direct-run/result.json`: corrected direct path.

The corrected fixture binary's SHA256 is
`04b4e3513de384b4354cf9842bbfaaf592a481e9975feaa10ec11bdc6a91a6c7`;
it contains 59 SM70 and 59 SM86 images. These synthetic-state checks do not
qualify inference arithmetic or turn fixture wall time into performance evidence.

For further movement checks, use the existing finite host fixture and shared
machine window with `STRATA_RESIDENT_DIRECT_COPY=0` or `1`; keep
`STRATA_TEST_PREFILL_SPLIT_OWNER` unset because it selects a different terminal
fixture. The current experiment contract in the project record,
`evidence/concurrency-campaign-20261016/bottleneck-trials-20261017.json`,
supersedes the earlier source-only instructions below: paired diagnostic
screens hold observer settings fixed, and promising candidates require a
reverse-order or held-out confirmation. Observer-off usefulness remains
necessary before promotion.


## Source-based memory price

Additional production model/device allocation: **0 bytes**. Additional pinned or
pageable payload/heap allocation: **0 bytes**, at every context and seat count.
Additional persistent owner/graph state: **0 bytes**. The existing per-thread,
per-device copy stream and its handle directory are reused, not duplicated.
Existing checkpoint clones and retained host pools are unchanged. The direct
branch omits the temporary `ConversationCheckpoint` (including its copied token
IDs) and per-layer `ConversationKv` host allocations; it does not reserve their
space elsewhere.

The observer adds exactly **12 counters × 4 uint64 fields × 8 = 384 bytes** of
fixed stack counter storage per scope, in either arm. Its compiled size bound
changes from 1024 to **1536 bytes**; a transfer record remains bounded at
**128 bytes**. Scopes are sequential on the native control thread, not per seat,
layer or retained request. No observer heap/device allocation is added, and the
off observer still performs no clocks, initialization of counters or logging.

Direct-copy metadata is fixed: two `ConversationStateSizes` (10 `size_t`s), up to
two `Layout` records (each compile-time bounded at 80 bytes), two five-pointer
pool arrays and one five-`size_t` array. These named objects total at most **360
bytes on the 64-bit target**, conservatively counting validation and transfer
locals together; scalar locals and compiler call frames are additional, not a
measured RSS claim. There is no context-sized stack buffer. No capacity cap or
12 GiB physical reserve may be reduced to admit this experiment. Parent must
retain ordinary monitor checks; bounded metadata overhead is within the approved
proportional acceptance scope, not grounds for daily/default promotion.

### Mode-specific host transient accounting (source only)

**Safety hold:** independent source review found a pre-existing retained-chain
undercharge; no OOM was observed. The source correction now prices each seat for
`L = max(prompt_cache, 1)` complete checkpoint owners independently of working/cache
aliases. Bind rejects a selected root/latest/all-pins chain above L before freezing,
allocating aliases or changing the old target; it never drops a pin to fit.
Flat stage topology, matching token IDs and stage payload shapes are validated first.
The boundary additionally checks actual retained storage before freezing: the five
nonempty array capacities, copied logical IDs, stage objects and future shared
controls must fit the per-checkpoint allowance computed by native admission.
That same precomputed allowance is passed to the owner; it is not repriced or
guessed at bind time. Capacity padding within the existing allowance is valid.
Caller-only ID/image/stage-directory reserve and empty mutable-array reserve are
not retained by the alias and are not a reason to reject an otherwise valid cut.
Shape alone cannot bound storage, but neither does storage require `capacity==size`.
Arrays above the actual admitted allowance refuse without spending opaque bytes.
For C checkpoint payload bytes, H seat-history bytes and S stages, the per-seat
checkpoint charge is `L * (C + ceil((C + H)/8) + S*4096)`.
Bind adds a serial alias peak A of `L*S*(4*cells + 4096)`, plus `ceil(A/8)`:
IDs and stage objects are newly allocated while the old target still exists;
the five sealed payload controls move/share payload storage rather than copying it.
Transfer and alias construction do not overlap, so admission uses their maximum,
not their sum. Histories remain `8*cells` per seat. Legacy image pricing and all
retention/settings stay unchanged. Offline planning stays in legacy-copy mode.
After swapping the prepared chain into the seat, bind clears old elements **and
destroys their top-level directory allocation** with an empty-vector swap before
`copy_to`. Thus no residual old directory is hidden in the serial transfer maximum.
During alias construction, old owners are covered by their persistent seat charge;
new ID vectors, flat stage objects and five shared controls per stage are covered
by A and the compile-time metadata assertion in `core/resident_memory.hpp`.
Eligible pins are counted in discovery, then distinct unpinned root/latest are
added; there is no third eligibility/prefix scan.
The CPU arithmetic and real-owner multi-pin/refusal regressions are source-only:
they have not been built or executed for this correction. Reduced-envelope model
runs remain unqualified pending parent review and those checks.

The full `resident_checkpoints` charge is now owned inside the existing
conversation envelope when prompt caching, a positive conversation budget and
positive parking slots are configured. Native admission saves the exact total and refuses, before
model allocation, if the envelope cannot contain it. `conversation_parked`
charges only the checked remainder; resident checkpoint bytes remain charged
in full. The cache is constructed before `ResidentSlots` (including the model
fixture return) and holds one non-evictable startup lease until teardown.
Publication refunds and poison clears do not refund that fixed ownership.
Disabled prompt caching, zero parking slots or a zero conversation budget keeps
checkpoint bytes external with no fixed lease.

For Fixed3GPU, 24 admitted elements at 137463862 B consume **3299132688 B**
of the unchanged **12884901888 B** conversation envelope, leaving
**9585769200 B** for parking, reads and other cache/store ownership. This is
reduced RAM parking/read headroom, not a quota increase, a freed allocation,
or a reduction in physical reserve, worker ceiling, context, prefill, slots
or groups. The standalone cache regression exercises real publication
handoff/refund, reservation eviction/refusal and poison-style clearing with
fixed ownership present; execution remains the parent's responsibility.

Source-only regression contracts: `tests/core/resident_model_test.cpp` constructs
separately unpinned root/latest and multiple pins with distinct ordered `used`
identities and stage-specific synthetic bytes using `CheckpointBytes::assign`.
Expected membership, pin flags, sizes and marker oracle exist before bind. Every
cut is freshly saved; the peer is independently saved, not a payload alias.
After input release, the real seat is checked against that oracle. The existing
ordinary-C++ allocation observer wraps cap-sized bind while the old target is
nonempty, and wraps unsealed over-cap, nested-stage, bad-shape and excess-capacity
refusals. Error capacity is reserved before observation, so refusal must allocate
zero observed bytes and leave all source payloads unsealed. Same-object metadata
representations, target/peer device snapshots, and retained contents must remain
unchanged. Completion resumes and admits the parked seat before cancellation/
finish; production lifecycle gates are unchanged. CPU arithmetic adds explicit
nonintegral ceiling and below/equal/above serial-max crossover constants in
`src/core/conversation_validation_test.cpp::resident_transfer_envelopes`.

Complete constructor inventory (source search of include/src/tests and SYCL,
after LSP references returned only the declaration): declaration/definition in
`program/resident_slots.hpp` / `src/program/resident_slots.cpp`; production
construction in `src/program/generate.cpp` supplies `max(o.prompt_cache,1)` and
the same precomputed per-checkpoint byte bound; `tests/core/resident_slots_test.cpp`
derives an explicit bound from the native pricing helper and supplies count six.
The model fixture borrows an existing owner, never constructs a second engine. No other constructor
caller occurs in those trees. Top-level `CMakeLists.txt` builds the same native
resident library for CUDA/HIP and links the generator and CPU preflight; the
allocation-observer vehicle is Unix/non-Apple and excluded for HIP. SYCL's
`strata_resolve` prefers migrated local sources and resolves its own generator;
no SYCL-local ResidentSlots constructor occurs. This is caller/source inventory,
not a claim that SYCL has native resident feature parity.

Parent verification only: build `conversation_validation_test`,
`resident_slots_test` and `strata-resident-test` in the approved configured build;
run CPU `conversation_validation_test` and CTest `resident_slots_preflight`.
The protected three-stage/four-seat resident-host fixture remains safety-held
until independent source review; do not run it as an ordinary CTest/model command.


The native preallocation adapter remains pure: its caller supplies the same
process-immutable selection later read by `ResidentSlots` and `MtpDrafter`.
Existing role-placement resolution/validation is unchanged. Both price modes
still validate main/draft image formats, geometry and checked byte arithmetic;
unknown placements, elastic growth and invalid geometry still refuse. Unknown
environment values select the unchanged conservative legacy path, not direct.

Allocation/lifetime audit:

- Legacy `copy_to` holds one fresh stage checkpoint while capturing/restoring
  one `ConversationKv` at a time. The checkpoint dies at the stage boundary;
  each image dies at the layer boundary. Main stages are serial. Draft save
  begins only after all main stages return, so the old bound remains
  `max(stage_checkpoint_peak + main_image, draft_image)`.
- Direct `copy_to` and live `copy_from` call `conversation_session_copy` without
  constructing either temporary owner. The session helper validates every
  source/target layer before writes and copies into the existing pools.
  Direct selected-cut restore copies KV first, then reads the retained host
  checkpoint and repairs the spare pooled row. These are borrowed payloads,
  not a new KV image; the retained-chain accounting correction above is separate.
- Both MTP slot directions call `conversation_kv_copy` in direct mode, never
  `conversation_kv_save`. Native resident KV, authoritative streamed host KV,
  hybrid K8V4, scales and draft rings use the same validated layout/pool helpers.
  `copy_payload` uses fixed metadata and bounded transfer chunks, not chunk
  allocations. Stream reset launches a kernel into existing map storage.
  Ring repair copies existing authoritative host runs into existing device
  slots and synchronizes; it allocates neither a host image nor a new ring.
- Format/extent/carve/device incompatibility returns false before copying.
  Transfer and ring-copy failures retain their existing failure handling;
  a streaming-reset launch error can terminate the native process. Neither
  outcome falls back to allocating snapshot code.
  The per-thread/device stream directory remains reused process-lifetime
  metadata. Runtime/driver internal staging is not proved absent and keeps its
  separate opaque allowance. True host-cache/store captures/restores remain
  image-backed and retain their independently admitted cache envelopes.

Only direct native resident transfer pricing passes zero serial image bytes to
the existing host-bound arithmetic. It **does not price zero transient**: it
retains `stage_checkpoint_peak + ceil(stage_checkpoint_peak / 8) + 4096`.
For the existing 262144-cell, split `20,39`, K8V4/MTP arithmetic reference, the
checkpoint peak is 50456596 B and the direct envelope is **56767767 B**, versus
the unchanged **335165271 B** legacy envelope (278397504 B less). These are
allocation-derived bounds, not an empirical RSS saving or a reason to fit
thresholds after refusal. Histories/checkpoints, startup scratch, physical
reserve, cgroup ceiling, cache budget, slots and model are unchanged. The
offline planner intentionally remains conservative at the legacy bound.

The existing host-only validation/transfer fixture now checks supported and
unknown main/draft formats, invalid context/splits/placements, overflow, retained
checkpoint/repair demand, and frozen selection across environment mutation.
Run it in separate processes with direct selection unset, `0`, `1`, and an
unknown value. Existing late-layer/failure/ring tests cover the strict no-fallback
consumer boundary rather than interpreting missing DMA counters as allocation
proof.

The three-stage/four-seat `strata-resident-test` vehicle now also wraps ordinary
C++ `new`/sized and unsized `delete` in its statically linked engine. A fixed
thread-local directory measures simultaneous newly allocated vector payload
bytes only while actual owner fresh, live, aligned/partial selected-cut,
invalid-cut, and terminal main/draft refusal operations run. It requires each
peak to fit the actual configuration's priced transient and prints
`RESIDENT_HOST_SCRATCH operation=... peak_cpp_bytes=... bound=... failures=0`.
Successful owner operations exercise main and draft transfers; invalid-cut or
main-refusal operations can stop before draft execution. The scope label is not
an execution attestation. The measure includes newly allocated metadata, but
excludes pre-existing owners, diagnostic snapshots, allocator bookkeeping and
driver-internal C allocations.
It is not total heap/RSS or GPU-compute telemetry. Directory exhaustion fails
the fixture, never silently undercounts. The observer is linked only into the
fixture; its directory scans add test-only overhead, so its timings must not be
used as performance evidence.

No new checks, builds or GPU/model runs have been executed for this accounting
change. Before using the reduced bound, the coordinator must run the minimal
protected real-owner vehicle in both modes, including the separate terminal
draft-refusal arm, and retain these scratch records alongside existing payload,
selected-row repair and cleanup checks. All earlier refused full-model runs
remain negative evidence. Neither this bound nor previous copy qualification
establishes full-model admission success, arithmetic quality, daily performance
or new HIP/SYCL qualification.


## Accounting and smallest acceptance checks

`STRATA_ADMISSION_COPY_BYTES` keeps its existing independent observer meaning.
The same scopes (`main_admit`, `main_resume_live`, `main_resume_cut`, `draft_admit`,
`draft_resume`) now support **`direction=direct`** alongside capture/restore.
Direct counts successful real copy calls once and is a logical direction, not a
PCIe measurement. Do not classify it as host traffic or silently drop it in the
parser. At most 36 part rows per scope are possible. Copy failures still record
zero successful bytes; incomplete scopes invalidate a performance estimate.

Parent owns all builds and GPU windows. Smallest useful validation sequence:

1. In this campaign, build affected CUDA SM70/SM86 engine and generation dependencies plus existing
   `conversation_validation_test`, `conversation_transfer_test` (CUDA GNU/ELF),
   `conversation_snapshot_test`, `kv_stream_parity`, and `strata-resident-test` targets.
   HIP and SYCL qualification is outside this campaign; neither build nor runtime
   support is established by this change. There are no new translation units.
   The earlier `CMakeLists.txt` change only adds `cudaMemcpyAsync` to
   the existing resident fixture's link wrappers so its fault injection reaches
   the current private-stream copy path; refresh that fixture's link options.
2. Run native validation/transfer fixtures: foreign carve, short destination and
   late-layer format rejection before CUDA/writes; first successful KV part
   followed by transfer failure with no fallback; one-leg byte count; pre-copy
   synchronization failure. Existing partial-write tests remain applicable.
3. Run native snapshot fixture: all F16/INT8/Q4/K8V4 source/destination residency
   pairs at 0/3/4/5/65 tokens, rotation mismatch, unchanged bytes beyond rounded
   extents, ring codes/scales, streamed-map reset, full GDN/PLE/index parity and
   selected-cut spare-row repair over a deliberately changed later pooled row.
4. Run the protected full-model resident fixture with the feature unset and with
   `1`, using the same approved startup/capacity parameters. Require payload and
   stationary-address roundtrip equality; consumer-drain refusals; exact yielded
   identity/seed retention; cancel/successor isolation; release/re-init; injected
   partial PLE-copy failure after GDN mutation; unaffected peer; working-resume
   poisoning. `STRATA_TEST_MIXED_RESIDENCY=1` additionally exercises the existing
   actual resident-working/ring-slot INT8 draft roundtrip, where supported.
5. Repeat the pinned coordinator client workload below with otherwise identical
   feature-off/on profiles. With the observer enabled, main fresh/live scopes
   must have direct parts and no capture/restore parts; draft scopes likewise.
   Selected-cut scopes retain restore-only running/checkpoint/repair traffic and
   direct KV. All exposed scopes must complete without failures. Compare state,
   tokens/usage, ownership and capacities, then measure observer-off latency in
   both arms. No numeric speedup or removal of all 783273504 bytes is asserted.

Use the existing `run_arm.py` cases, fixture hashes, monitor and window wrapper:

```json
[
  {"kind":"mixed","name":"direct-slot-copy-cold"},
  {"kind":"cancel","name":"direct-slot-copy-cancel-successor"}
]
```

The mixed case submits `active-decode`, waits for its first visible content, then
submits `cold-prefill`. The cancel case closes `active-decode` after two visible
events and submits `negative-live`. Keep the existing warmup, four 262144 seats,
two groups, 2048 prefill, 24576 experts and model. Parent pins each candidate
binary/hash/profile and plan before running its approved `window.py` invocation.
These cases alone do not establish cached/cut restore exposure: the existing
retained-history fixture must emit actual live/cut resume and draft scopes, or
those surfaces remain unexercised. Keep the current 12 GiB reserve and thermal/resource stops.

Remaining assertions are compilation, real-device fixture success, actual
copy-direction/count reductions, lifecycle/state equivalence and measured
observer-off usefulness. Direct-to-slot prefill remains unimplemented because
this narrower candidate removes the intermediate snapshots without changing
ownership. No publication, push, hardware claim or default promotion is made.

## Source-only review correction, 2026-10-14

Correction base: `2660ee47513ac21b6c85956c900d22b74c085e8f`. No build, test, lint,
formatter or GPU command was run for this correction. LSP reference requests
returned declarations only (no usable project index); repository-wide source
search supplied the caller inventory. These are source findings, not passing
test results or measured speedups.

The confirmed software-error chain was `ResidentSlots::bind` →
`MtpDrafter::save_slot` → `conversation_kv_copy` (or snapshot restore) →
`restore_residency` → `kv_ring_restore`. The ring mover discarded the
`cudaMemcpyAsync` return value and called `check`, which read `cudaGetLastError`.
A link-wrapper refusal returns `cudaErrorInvalidValue` without populating that
runtime slot, so both `check` and the later `restore_residency` read could report
success after only part of the ring had been copied. This is a demonstrated
source-control-flow defect; whether a particular real CUDA driver failure
produces the same empty-slot condition remains untested. NVIDIA documents the
last-error read and its reset separately in the
[runtime error API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html).

`kv_ring_restore` now returns a checked boolean plus the actual failed submission's
error text and stops at that submission. The shared declaration, CUDA/HIP source,
native snapshot and MTP callers, both parity callers, and SYCL implementation and
callers use this contract. SYCL reports a caught submission exception through
the same explicit failure result and message; it does not convert it to success.
The native boolean MTP restore path returns failure, so an unfinished `Operation`
retains its poisoned scratch lease. The existing void `kv_restore` entry remains
fail-stop. Synchronization errors still prevent successful publication; successful
submission alone is not treated as completed device work.

For direct main admission/resume, `conversation_session_copy` now owns the entry
device fence. `ResidentSlots::idle` still checks consumer ownership, and `OnDevice`
still selects each stage's device. Session validation uses geometry, extents and
pointers only, performs no transfers, and completes for every layer before that
fence. Removing the caller's earlier fence therefore does not move any payload
access before the remaining fence. The default snapshot branch keeps its previous
fences. Direct timing now includes this entry fence in `main_direct_transfer`
instead of emitting a separate `stage_sync` row. Completion and selected-cut
checkpoint fences are unchanged.

Standalone KV copy now reuses the source layout produced by pair validation.
Session copy keeps all-layer prevalidation, then uses a private second-pass helper
that recomputes only source byte counts; it does not repeat both owners' layout
and pointer validation. Its contract requires joined consumers and stable owner
metadata through the copy. There is no per-layer plan array or extra payload
allocation. Selected-cut KV-before-checkpoint repair, 16 MiB authoritative-copy
chunks, stationary owners, capacities, expert residency, model arithmetic and
the exact-`1` switch are unchanged. The ring error fix applies to both switch
settings; successful off-path payload operations are source-equivalent, but
byte-identical output has not been measured here.

### Coordinator verification strategy (not executed)

Use the already approved CUDA SM70/SM86 build configurations and protected GPU
window. In the commands below, `campaign_build` names the coordinator's configured
build directory with `STRATA_BUILD_CONVERSATION_TESTS=ON` and `STRATA_BUILD_TESTS=ON`:

```sh
cmake --build "$campaign_build" --target strata conversation_validation_test conversation_transfer_test conversation_snapshot_test kv_stream_parity strata-resident-test
CUDA_VISIBLE_DEVICES=-1 ctest --test-dir "$campaign_build" --output-on-failure -R '^conversation_(validation|transfer)_test$'
```

The minimal failing-before comparison is to apply only the changed
`src/core/conversation_validation_test.cpp` to the correction base and run
`conversation_transfer_test`. The new ring fixture uses ordinary host buffers
and wrapped CUDA calls, refuses V after the first ring K submission, and keeps
the last-error read successful. It covers snapshot and direct repair in F16,
INT8 and Q4 with a wrap boundary. Before the fix, it should fail the actual-error
assertion because restore returns success; after the fix, it must stop at the
refused copy with the first ring page partially changed and the wrapped page
untouched. No result from either arm is claimed here.

The late-layer test now uses independent source and destination allocations with
`0x3c` and `0xa5` contents, different PLE history, and corruption of either the
last source or last target layer. It checks unchanged destination bytes and zero
copy, device-sync and stream-sync calls. The base already prevalidates all layers,
so this coverage improvement should pass both correct implementations. To check
the fixture's sensitivity, a coordinator-only mutation that removes the first
all-layer validation pass must fail it on an early write or synchronization.

For real owners, run the existing protected `strata-resident-test` command with
its approved model/capacity/startup arguments in a profile that has ring draft
slots. Add `STRATA_TEST_RING_COPY_REFUSAL=1`, once with direct copy unset and once
with `STRATA_RESIDENT_DIRECT_COPY=1`. The existing mixed-residency profile can
also retain `STRATA_TEST_MIXED_RESIDENCY=1`. This terminal fixture marks incoming
K/V and destination ring bytes differently, refuses the actual ring V address,
and requires `RESIDENT_RING_REFUSAL ... poisoned=1 rebind_refused=1 reuse_refused=1
failures=0`. It asserts partial K mutation, unchanged V, no target publication,
failed working/slot rebinding, failed save/restore reuse, rejected successor
admission, and unchanged peer fingerprint/identity. It ends there because poison
must persist. Run the ordinary fixture separately with the new refusal variable
unset to retain both existing PLE-failure cases.

Under the same protected window, run `conversation_snapshot_test` and
`kv_stream_parity`, then the pinned off/on client workload above. Remaining risks
are compilation across changed declarations, CUDA wrapper interception from the
kernel library, actual asynchronous/default-stream repair ordering, poisoned-owner
destruction, selected-cut and full-model parity, and latency on both CUDA targets.
HIP and SYCL declarations/callers were migrated from source only; neither backend
was built or executed. No deployment, binaries, model files, other checkout,
parent gitlink, push or default promotion is part of this correction.

## Current host fixture seam (not yet exercised)

The legacy resident executable's protected, single-device/two-slot entry cannot
qualify the campaign's three-device/four-slot host profile. Its test-only host
entry now retains that real allocation and two execution owners rather than
faking protected-supervisor state or shrinking capacity. Each stage and the draft
are captured on their own device. It covers all-stage payload roundtrips,
stationary addresses, live and selected-cut resume, yielded identity/seed,
consumer retirement, cancellation/successor reuse, partial main admission,
unaffected peers and poisoned working resume. A separate terminal
`STRATA_TEST_DRAFT_COPY_REFUSAL=1` case injects refusal after draft K mutation and
requires draft scratch and successor reuse to remain poisoned.

The host case uses the canonical resident draft allocation. It does not replace
the separate ring-copy helper fixtures or claim a real-model ring-residency run.
It admits a conservative eight-image upper bound inside a 2 GiB diagnostic host
allowance, with at least 8 GiB physical reserve remaining; it adds no GPU payload.
Run it only through the normal host resource/thermal monitor and owned MPS
ceilings, in a finite measure window. Production builds exclude the fixture entry
and stage-directory copy. Compilation and execution of this host seam are pending;
the already recorded CUDA helper passes are not a pass for the new lifecycle case.

The first host fixture run stopped at an invalid inherited test premise: retaining
only the global expert ledger does not create a verifier row consumer when two
execution owners are enabled. `slot_idle()` intentionally permits retirement of
an unborrowed row while other work exists. The unsupported per-row refusal
assertion was removed, not treated as an engine defect. Global admission's
retained-consumer and cancelled-but-undrained refusal checks remain; cancellation
of an actually submitted GPU row still belongs to the real-client scenario.
The failed fixture and clean native/MPS teardown are retained as negative evidence.
