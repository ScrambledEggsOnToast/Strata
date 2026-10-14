# Immutable prefix backing candidate

Source-only candidate on `0567f3039aadf07fdf6462652e71878d365575d3`, in the
isolated `generation-prefix-extents-20261014` checkout. No compilation, tests,
lint, formatters, downloads, GPU work or services were run for this change.
The parent owns all builds, executable validation and publication. Local commit
hooks are bypassed because this dispatch forbids their builds/tests. The earlier
discriminator record below is historical and describes the base implementation.

## Scope and control

`STRATA_PREFIX_BACKING=1` enables 64 KiB host segments and actual backing reuse
between authorized snapshots. Unset or `0` keeps the original 16 MiB payload
segmentation and disables sharing. **Use the original binary as the control:**
this candidate changes `ConversationBuffer` and segment directory representation,
including their `sizeof` and accounting, even with the flag off. A candidate-off
comparison is useful for diagnosis but is not an uncontaminated original control.
All dependent objects must be rebuilt; do not mix old/new objects across this ABI.

This is explicitly a **retained-allocation** optimization. It captures the full
payload first, checks the full eligible cut, then releases duplicate whole host
allocations by pointing to existing backing. It does not relocate a prefix or
private tail into another allocation. It removes **zero capture or restore
transfer bytes**, does not reduce capture peak, and does not add GPU payload.
Existing active-buffer reuse still has its separate `reused_kv_bytes` meaning.
Smaller segments increase transfer call count; no throughput/latency benefit is
claimed. No run, quality, speed or measured memory result is claimed for this code.

The supplied prerequisite reports 25,358,208 equal eligible bytes at cut 2048 in
epochs 3/4 of the authorized-sibling discriminator. That is prior evidence, not a
run of this candidate or its expected unique saving. Reuse is rounded down to
whole matching allocations, so small scales, pooled arrays and partial extents
can remain separately backed. Never sum the pairwise duplicate witnesses.

## Identity and ownership

The serving caller validates completed recurrent payloads and every stage against
the loaded session before invoking the sharing helper. The helper reuses the
existing probe's exact authorization scope, steering, geometry, layer carve,
format, token/image prefix and common-checkpoint checks. Model and arithmetic
identity remain fixed for the cache lifetime. Each stage must contain the same
real checkpoint; identical token text alone never creates eligibility. Stage
steering is also compared. No root or recurrent payload is fabricated.

Only exact-capacity 64 KiB segments entirely inside the eligible prefix can be
shared. Both source/receiver segment boundaries must match and the bytes are
compared again before replacement. Main pages, conservative draft pages ending
before `cut-1`, optional K/V scales and completed indexer rows use the existing
format arithmetic. The partial page, incomplete draft page, moving indexer row,
subextent payloads and every mutable tail remain private. Unequal bytes anywhere
in the complete cut reject replacement of every earlier layer/stage too.

The first complete equal donor wins, deterministically in oldest parked order;
only then is a held incoming considered. There are at most eight inspected
sources and 64 MiB of full-cut comparison work per capture, plus at most another
64 MiB of full-extent rechecks. Coverage limits may yield no sharing. The old
optional observer has its own unchanged 128 MiB/eight-pair budget.

Each shared allocation has reference-counted lifetime. Eviction, `take()`,
active `retain()`, dirty-prefix invalidation and failed capture can release one
reference without dangling another. Ordinary buffer copies remain deep copies.
A mutable visitor or partial shrink detaches an aliased segment before exposing
its writable address. Allocation/copy finishes before the owning pointer changes;
a failed allocation leaves all aliases intact. Mutable visitor pointers must not
outlive another mutation/sharing operation; audited production users perform
synchronous copies and retain no writable host pointer.

## Admission and host metadata

All aliases are charged in full by existing `bytes()` accounting, so the cache
gets no extra admission credit from sharing and the same maximum seats, expert
residency, cache budget and host reserve remain required. For every shared
segment, `allocation_peak()` additionally charges a full replacement allocation
and owner, including for same-size rewrites and shrinking. Directory growth counts
both old and new directories. The existing CUDA/HIP and SYCL capture estimators
already consume this method before the retained-buffer path can mutate storage.
The physical gate subtracts retained charged bytes only after this peak has been
priced. Evicting a donor may reduce the eventual detach cost; no such speculative
credit is taken before admission.

Derived host charges, not measurements: each segment directory element contains
one vector and one shared pointer; each opt-in owner adds one vector plus two
pointer widths, following `CheckpointBytes`' existing control-block convention.
On the usual 64-bit standard-library ABI these are 40 bytes for the directory
entry and 40 bytes for the owner, per at most 64 KiB of fresh payload, with one
partial tail per buffer. `sizeof` is used in admission, not hardcoded ABI sizes.
The buffer adds its policy flag/padding (normally 8 bytes), included through
`sizeof(ConversationKv)` in snapshot estimates. Vector capacity is charged.
Allocator bookkeeping is excluded under the existing accounting convention.
The session-file preflight now prices a page per 64 KiB segment when enabled,
retaining its read buffer and small-vector allowance. Shared extent release frees
live allocations to the host allocator; it does not promise an immediate RSS drop
or OS page return. Physical admission continues sampling actual available memory.

## Diagnostic and acceptance

Set `STRATA_PREFIX_BACKING_PROBE=1` separately to extend the existing
`PREFIX_EXTENT_PROBE` stream with `phase=backing-retained` and, when applicable,
`phase=backing-with-held`. `STRATA_PREFIX_EXTENT_PROBE=1` still enables the original
byte-equality/alias pair witnesses. Both diagnostic flags perturb timing.

The new diagnostic prices `sizeof(PrefixBackingLedger)` before capture in both
cache and physical admission. Its temporary heap directory contains at most
8192 allocation pointers (64 KiB plus counters on 64-bit); it retains no payload
references or registry after the park. It visits at most 8192 segment references,
using exact pointer identity to count each live allocation once (plus the first
over-limit reference that triggers truncation). `complete=0`
invalidates the totals as full coverage; it must never be read as zero savings.

For `complete=1`, `logical_payload_bytes - unique_payload_bytes` equals
`alias_payload_bytes`, the payload allocations avoided within that simultaneous
owner set. Owner/control metadata is separately reported as logical/unique bytes;
`charged_buffer_bytes` includes all buffer directories, capacities and fully
charged owners. Thus directory bytes equal charged minus logical payload minus
logical owner bytes. `cache_charged_bytes` also contains other snapshot metadata.
The parked-only phase excludes the transient incoming owner. The with-held phase
adds it to the same ledger, without counting previously seen pointers twice.
Active reuse has been taken out of the cache and moved into this completed
capture at this observation point; it is not an omitted live owner. Explicit
`capture_copy_credit_bytes=0 restore_copy_credit_bytes=0` prevent transfer claims.
Allocation bytes are not OS working-set measurements.

Added behavior fixtures in existing CPU executables cover: three sibling aliases
and unique counts; untouched tail addresses; private draft final pages; priced
same-size/partial-shrink detachment; mutation isolation; donor eviction, take and
active reuse; scope, steering, geometry, carve, checkpoint and payload rejection;
all-stage eligibility; late-layer byte mismatch; comparison/diagnostic limits;
exact frozen session-file bytes with actual shared backing plus load mutation;
and allocation-failure injection for detach writes/shrinks with unchanged donors.
These fixtures are authored, **not run**. Existing failure/restore tests remain
part of the parent's validation. After rebuilding all touched backend closures,
the exact finite CPU commands are:

```sh
: "${GEN_PREFIX_BUILD:?coordinator build directory required}"
env CUDA_VISIBLE_DEVICES=-1 ctest --test-dir "$GEN_PREFIX_BUILD" --output-on-failure --timeout 120 -R '^conversation_(cache|file|validation|split_failure)_test$'
env CUDA_VISIBLE_DEVICES=-1 STRATA_PREFIX_BACKING=1 ctest --test-dir "$GEN_PREFIX_BUILD" --output-on-failure --timeout 120 -R '^conversation_(cache|file|validation|split_failure)_test$'
```

Use the original binary for the measured control. Freeze a new candidate plan
for the same root/B/C sibling fixture, same scope and resource settings, with
`STRATA_PREFIX_BACKING=1`, `STRATA_PREFIX_BACKING_PROBE=1` and
`STRATA_PREFIX_EXTENT_PROBE=1` in its private profile. Keep longest-prefix
selection and the 2048 checkpoint/prefill partition. Invoke the existing bounded
driver command below with that new plan; do not modify the frozen old plan.
Require complete unique-allocation witnesses, positive retained aliases,
unchanged successful response/cached-token behavior, scope isolation and clean
teardown. Compare total unique backing plus metadata against the original under
the same offered requests. This source candidate is not qualified until the
parent executes that work; host RSS and latency remain separate observations.

## Source audit and build closure

Audited mutable/const buffer consumers: CUDA/HIP and SYCL snapshot capture,
restore and verification; retained snapshot state; file writer/reader; cache
ownership and accounting; resident-slot/MTP temporary captures; and existing
CPU/GPU test callers. Resident/MTP scratch uses fresh private images, never a
shared donor; its existing `resident_host_bytes` 12.5% plus 4 KiB transfer
allowance covers the derived per-64-KiB metadata increment. Const serialization preserves ordering and bytes, while parsing
builds private writable buffers. No disk schema or device allocator is changed.
The serving integration is in the CUDA/HIP `src/program/generate.cpp`; SYCL
shares the representation/estimators but has no new sharing integration.

Clangd LSP reference requests were used for the exported buffer type,
`allocation_peak`, `share_prefix`, `for_each_backing`, `share_prefix_extents` and
`PrefixBackingLedger`, with background indexing and clang-tidy disabled. An
initial inherited include-path mismatch was corrected with a temporary local
compilation database (removed after navigation), without running a compiler.
The resulting open-file index resolved 8/4/2/9/5 references respectively; it was
supplemented by repository searches and is not a complete backend-index validation.
No compilation/test/lint success is implied. Rebuild all consumers of
`conversation_buffer.hpp`, including file/state/snapshot, resident/MTP and tests,
plus `generate.cpp`; the parent must build CUDA, HIP and SYCL before review.

### Coordinator correction: transactional partial shrink

The independent spec review found that a failed detach during a multi-extent
shrink could discard the receiver's private tail before allocating its replacement
prefix. A queued native CPU reproduction observed `before=131072 after=65536`
after the injected allocation refusal; the donor remained intact. The correction
detaches the surviving partial extent before removing any trailing extents.
The same reproduction then reported `before=131072 after=131072`, and the existing
allocation-failure executable passed with one- and two-extent receivers, checking
original size, charged bytes, alias identity and all receiver bytes on refusal.
These checks establish this CPU failure boundary only. They do not establish
full-model sharing, memory savings, backend qualification or default readiness.

---

## Historical discriminator record (before backing reuse)


Source base: `f26653cbaea52a8d9e687807d3e024d6e2ef50f6` in the isolated
`generation-prefix-extents-20261014` checkout. Implementation: `5cf25c11`.
No builds, tests, formatters, downloads, GPU work or services were run. The original
`generation-prefix-20261014` checkout was not changed. Commit hooks were bypassed
because this dispatch explicitly forbids their builds/tests.

**Delivered scope:** the executable discriminator explicitly allowed by the brief.
Shared KV storage, copy-on-write tails and a physical-allocation ledger are **not
implemented**. No absence-of-gain or memory-saving result is claimed. The observer
measures real equal, separately backed prefix bytes at actual parking points; it
is not a standalone hypothetical byte model or a checkpoint-sharing implementation.

## Why use the discriminator first

`ConversationBuffer` owns mutable vector segments of up to 16 MiB. A 2048-token
prefix commonly ends inside a segment containing a branch's private tail. Sharing
that vector would share writable tail bytes; retaining just a pointer into it also
retains the whole allocation. Actually reclaiming the receiver's prefix would
require new ownership, splitting/rebuilding its private tail, and accounting for
sources retained after eviction and failed copy-on-write allocation. Changing
only a pointer type cannot meet that contract.

Existing `take()` removes the restored parent from the parked cache, `retain()`
moves its uniquely owned KV into the active reuse buffer, and `drop_superseded()`
can remove a sibling before the next capture. Therefore a common token root does
not establish two simultaneously retained immutable allocations. The discriminator
checks precisely that prerequisite without widening scope, altering retention or
flattening KV buffers. This is a reason to measure, not proof of no gain.

## Invocation and frozen setup

Enable `STRATA_PREFIX_EXTENT_PROBE=1` in the coordinator's private profile. Unset
or `0` skips all comparisons and witness output. This adds no members to the
buffer/cache representation, changes no byte accounting, and leaves all existing
transfers and eviction decisions intact. The only added cache API is synchronous
const iteration. Off-path equivalence is established by source inspection only;
no executable comparison has run.

Use the same source artifact for observer-off/on. Hold
`STRATA_IMMUTABLE_CHECKPOINTS`, `STRATA_SNAPSHOT_DIRECT_CUT`, model/steering, scopes,
cache budget/slots, four full-context seats, full expert residency and 2048 prefill
partition fixed. Use `--prefix-selection longest`: observed park time includes the
probe and must not feed a cost-sensitive treatment comparison. Keep
`--prompt-cache-every 2048` fixed to establish the real common checkpoint.

Start with the frozen `history-fixtures.json` and one existing `kind: history`
case in the coordinator's `run_arm.py` plan: root, four distinct fillers,
parked branch, live repeat, different scope. Pin the fixture, frontend, executable,
profile and driver hashes in a new create-only plan. The brief records root2304 /
branch2305 sharing 2280 tokens and a real 2048-token retained cut; this worker did
not rerun that evidence. Do not silently alter the public fixtures to force a hit.

After the coordinator creates and pins `GEN_PREFIX_PLAN` to that actual plan, the
existing driver accepts exactly this invocation (not executed here):

```sh
: "${GEN_PREFIX_PLAN:?new pinned observer history plan required}"
python3 /home/joshkirklin/ai-optimise/tools/host-inference/window.py run --class measure --wait-s 1800 --max-s 1800 -- env GENERATION_MEASURE_WINDOW=measure python3 /home/joshkirklin/strata-workspace/scratch/generation-campaign-20261014/run_arm.py "$GEN_PREFIX_PLAN"
```

This driver uses the existing `measure` helpers and normal host launcher/monitor.
Use its finite profile with timeout no greater than 1800 seconds, unchanged 8 GiB
reserve and resource/thermal stops. No model/GPU work may overlap another holder.
Stop at the first failure; do not rerun a safety-refused configuration.

If the frozen sequence yields no simultaneously retained sibling pair, report that
limited observation. A separately frozen extension may park two branches whose
own deeper checkpoints prevent superseded removal while retaining the common root.
Tokenize and freeze it before execution, preserve the same authorization scope,
and price its extra requests; do not call an unrun extension qualified.

## What the witnesses mean

Each successful capture may emit `PREFIX_EXTENT_PROBE` records, keyed by `epoch`:

- `phase=held-incoming`: the outgoing capture versus the incoming snapshot already
  removed by `take()` and held for restoration. This is simultaneous **transient**
  backing, not evidence that both images remain parked after the exchange.
- `phase=retained`: the newest successfully published snapshot versus older entries
  **after** superseded removal and eviction. Each pair really remains parked at
  the observation point. A borrowed reference never outlives synchronous iteration.
- `phase=summary`: `stored`, current `parked` count, attempted pair count capped at
  eight, truncation flag and total compared bytes. Zero pairs is explicit, rather
  than silent missing exposure. Parking skipped before capture emits no summary;
  its existing skip/failure log remains authoritative.

Scope and steering mismatch returns `status=scope` before reading payload or
validating the foreign image. Otherwise the actual serving path validates every
source/stage with `conversation_snapshot_validate` before probing. Invalid state
returns `source-invalid`; observer allocation failure returns `allocation-failed`
and does not change whether the captured image is published.

The pure comparison helper requires the same geometry/layer ranges, formats,
shape metadata and an actual common checkpoint with matching token/image prefix
in **every stage**. It searches cuts up to 4096 tokens, at most 64 checkpoints and
64 KV layers per stage, and at most eight extra stages. It uses the existing
`qsa_image_bytes` format arithmetic, including current K8V4 main/INT8 draft,
rotated formats and optional scale arrays, rather than a bytes-per-token guess.
The helper relies on its serving caller's recurrent-payload validation; its CPU
fixtures exercise structural selection and buffer comparisons only.

For cut C and page size P, main K/V/scales include only
`floor(C/P)*P` cells. Draft includes only `floor((C-1)/P)*P` cells, conservatively
excluding the potentially uncomputed final cell's whole page. Pooled index keys
include `floor(C/idx_block)` completed rows; the moving spare row is excluded.
GDN, PLE, tails, indexer running state, token metadata, allocation slack and vector
bookkeeping are never included in `eligible_bytes` or `duplicate_bytes`.

Const buffer visitors compare bytes directly in chunks no larger than 64 KiB;
there is no hash collision assumption or flattening allocation. Only `status=equal`
certifies the **entire selected eligible extent**. `duplicate_bytes` counts equal
bytes at different addresses; `aliased_bytes` counts identical addresses, which
are not savings. In the current uniquely owned vector storage, different backing
addresses identify independently allocated payload. `different` zeroes both
opportunity fields; partial comparisons are not a complete-cut witness.

`limit`, `cut-limit`, `layout`, `source-invalid`, `allocation-failed`, truncation,
and `no-cut` are refusals or coverage limits, not measured zero-gain findings.
`no-cut` includes a pair without any structurally compatible bounded cut.
There is a 128 MiB comparison budget and eight-pair cap per capture. The frozen
8-entry cache plus one held incoming fits the pair cap. Summaries count inspected
pairs only; a budget-limited record makes completeness unavailable.

Do not sum pairwise duplicate bytes over epochs or peers as a physical-memory
saving: the same root can occur in many pairs. A single positive retained pair
establishes duplicate immutable payload at that instant. Reclaimable allocation
bytes still depend on segment splitting, tail copying, ownership and retirement;
there is no H2D-saving claim because destination GPU arenas remain private.

## Observer cost and finite assertions

This is functional-only observation. It reads both host copies, may fault pages
in, validates metadata, emits stderr and adds watchdog beats. Park timing and host
memory traffic are perturbed; no speed comparison can use these observer-on
numbers. No GPU work or additional transfer stream is introduced. Snapshot sources
are drained before capture publication and remain alive throughout the probe.

Added assertions live in the existing `conversation_cache_test` executable:

| Scenario | Assertion |
| --- | --- |
| K8V4 main + INT8 draft, nine captured tokens, real cut five | 968 separately backed eligible bytes; no tail or recurrence bytes counted |
| Mutate partial main/draft page and moving index row | Duplicate witness remains 968 bytes |
| Mutate an eligible KV byte while token IDs still match | `different`, zero complete-cut opportunity |
| Different tenant scope or steering | No KV bytes inspected |
| Remove actual checkpoint or one stage's common cut | No eligible cut manufactured from tokens |
| Malform the final draft buffer | Refused before comparisons |
| One-byte comparison budget | `limit`, zero inspected bytes, not a zero-gain result |
| Compare an image with itself | Aliased bytes counted, duplicate bytes zero |
| Page-aligned cut four | 440 bytes; draft last page remains private |
| Two complete stage images | 1936 bytes, each stage counted once |
| Observe without mutation | Contents, backing and charged bytes unchanged |
| Different segment boundaries, including a third segment | Equal prefixes compare equal; an eligible third-segment mutation invalidates the witness |

Once built under the shared build window, the exact finite CPU command is:

```sh
: "${GEN_PREFIX_BUILD:?coordinator build directory required}"
env CUDA_VISIBLE_DEVICES=-1 ctest --test-dir "$GEN_PREFIX_BUILD" --output-on-failure --timeout 120 -R '^conversation_cache_test$'
```

No assertion was executed here. Actual-path acceptance additionally requires the
expected parked checkpoint/live-hit/different-scope legs, complete offered requests,
unchanged residency and clean teardown. A cancelled request and successor can use
the coordinator's existing bounded cancellation case; no new cancellation or
allocation-ownership implementation is introduced by this observer.

Independent review found that both the new comparator and the existing buffer
visitor added an absolute offset before subtracting the segment origin. The
corrected expressions subtract first, keeping intermediate pointers within the
allocation. The added segmented fixture covers equal and unequal payloads across
different partitions; it is not a claim that this C++ undefined behavior reliably
failed before the correction.

## Files and build closure

Changed: `include/strata/core/conversation_cache.hpp` (const visitor only), new
`include/strata/core/prefix_extent_probe.hpp`, `src/program/generate.cpp`, and
`src/core/conversation_cache_test.cpp`, plus this report. No CMake/new translation
units or ABI representation changes. Recompile `generate.cpp` and the cache test;
the normal depfile closure for `conversation_cache.hpp` also includes conversation
file/snapshot/state, MTP, resident slots, serial judge and their dependent tests.
Relink affected executables and preserve the base's immutable-checkpoint ABI.
CUDA/HIP/SYCL builds remain the coordinator's obligation, not claimed passes.

Remaining work is explicit: execute the discriminator once on the frozen history,
interpret retained versus transient coverage, and decide whether a prospective
sibling extension or a full immutable-segment ownership implementation is justified.
Missing measurements remain missing evidence. Shared prefix storage is unimplemented.
