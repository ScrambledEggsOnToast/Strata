# Retained-layer prefill experiment

Source baseline: `335b954daa473fead32e17acd9e7e898944115f5`.
This is source implementation and static reasoning, not build, runtime, quality,
or performance evidence. No builds, tests, formatters, GPU work or services were
run in the implementation wave. Read alongside the separate
[stationary ownership evaluation](stationary-prefill-evaluation.md).

## Event and lease comparison

A bulk owner retains its arena, request binding, token span, layer cursor,
recurrent ordinals, PLE producer, expert plan, residual and fused normalization
buffers. Device execution availability is a different resource. A verifier group
retains replay scratch and handoff rows until accepted-prefix commit and draft
advancement; a completed stage alone does not release that group.

| Policy | Device becomes available to decode | Bulk resumption condition |
| --- | --- | --- |
| Global closure | Entire admitted prompt span ends | Every group drains before the next bulk span |
| Stage-only | Complete bulk stage ends | Next stage acquires its device; no interior stage cut |
| Retained layer | Existing bulk layer completes locally | Only the next bulk device's running verifier stage drains |

A concrete counterexample rejects the premise that stage-only release captures
all possible benefit. Assume three devices, two bulk layers of four time units
per device, one decode window with one unit per stage ready at time 1, zero
handoff/commit costs, and unchanged service demands. Global closure publishes
at 27; stage release can publish at 25. With the implemented callback cadence:

| Time | Event |
| --- | --- |
| 0–4 | Bulk stage 0, layer 0 |
| 4–5 | Decode stage 0 while the bulk frame is retained |
| 5–6 | Decode stage 1; bulk stage 0 layer 1 runs 5–9 |
| 9–10 | Next callback observes stage 1 and launches decode stage 2 |
| 9–13 | Bulk stage 1 layer 0 |
| 13 | Callback observes decode stage 2 and publishes after commit/draft join |

Bulk completion moves from 24 to 25, even with zero switch cost. This is a
hand-derived reachability witness, not a simulation run or a hardware forecast.
It establishes a possible response-gap advantage and a possible TTFT cost.
An ideal continuously polling event scheduler could publish earlier; this
implementation deliberately does not claim that scheduler's bound. Actual
layer, commit, draft and transfer demands must be measured independently.

Adversarial schedules do not change the ownership rules:

- A late decode completion retains its group and prevents another launch on
  that device until `batch_poll` returns success. Other devices can advance.
- A completed early stage retains deferred scratch while a late final stage
  determines acceptance. The existing `commit_slot_prefixes` chain remains the
  group join; it is not replaced with an early release.
- Cancellation marks intent. Resident finish still uses `slot_idle`, including
  both verifier owners and the draft owner. No new admission runs in a layer
  callback; contiguous controls cannot cross a queued admission.
- A failed prompt keeps the existing poisoning behavior. Unwinding retains
  buffers until local drains complete; the caller drains remaining verifier
  groups before returning to admission/error handling. Failure is not completion.

## Implementation and isolation

`STRATA_PREFILL_RETAINED_LAYERS=1` selects the generation-only experiment.
`STRATA_PREFILL_RETAINED_LAYERS=stage` is the matched stage-only control: identical
serial handoffs and leases, but no interior layer callbacks. Compare both with
flag-off global admission; stage-only is not a claim of reproducing #227 scoring.
Unset or `0` leaves `run`, asynchronous successor submission, chunk policy and
pipeline launch policy unchanged (the extra blocked-stage check is always false).
No default, model, routing, quantization, sampling or context capacity changes.
Four request states and two verifier groups remain allocated. SYCL explicitly
refuses the flag. CUDA and HIP share the implementation but neither was built
or exercised in this wave.

`Prefill::run_retained` validates an owned 2048-row arena on each distinct device,
resident KV, complete expert residency and no helper/CPU-share borrowers. The
serving guard also refuses adaptation, borrowed scratch, peer experts, split
speculation and the old foreground-budget mode. Tail chunks keep their original
row count. Each existing prompt span is partitioned into the same 2048-row chunks.

The coordinator's first stage-only smoke refused before serving a request:
the startup guard required the requested `no_prefill_borrow` setting even though
all three actual arenas were owned and all 24576 experts remained resident.
The corrected guard checks resolved loans and each actual 2048-row arena instead.
It changes no allocation, partition or residency setting. The original refusal
and clean unit teardown are retained; a corrected runtime result is still pending.

The continuation is the live C++ call stack. The layer bounds never change;
`normed`, `R`, `grs`, `xn16`, optional `xn16_lo`, staging indices and ordinals
remain where the original loop put them. No layer return re-enters preparation,
changes fusion, reacquires a request canary or allocates another device arena.
Compute/copy/KV side streams join before the callback. Decode has its existing
separate replay arena; shared PLE scratch is only reused after the bulk layer
completes and is no longer a next-layer input. No new GPU events or allocations
are introduced; callbacks and a retained pointer are the new host state.

Bulk stage handoffs are serial in this mode. The producer frame and its pinned
handoff stay alive until the descendant returns. Entry callbacks acquire the
next device before any prompt submission. Descendant callbacks may lease an
ancestor's device to decode, so ancestors reacquire it before their epilogues.
Entry acquisition polls decode before waiting on a shared stream. Ordinary
interior boundaries synchronize only their local bulk producers.

The callback offers one pipeline pump, then prevents launches on the returning
bulk device while polling its current stage to completion. It does not close a
whole generation round. Commit/draft remain synchronous existing operations;
this is not an asynchronous commit rewrite. The existing global admission fence
and working-to-resident copies remain at prompt-span completion. BYIELD is not
consumed by the new layer callback; existing pipelined admission behavior remains.

## Bounded assertions and coordinator validation

No assertion below is reported as runtime-passed. Stop at the first functional
failure; do not repeat an unchanged failing candidate.

| Scenario | Assertion / existing observation |
| --- | --- |
| One 2048-row prompt plus final tail, off/on, no peer | Same chunk boundaries, fused path and final GDN/PLE/KV/draft state; use existing state fingerprints and residual dumps as local byte diagnostics |
| One active decoder plus one fresh 2048-row prefill | `RETAINED_PREFILL` shows an interior next-layer cursor, local idle and another group still in flight; committed output can occur before bulk completion |
| One 2049+ row prompt | Successor handoff and next-chunk reacquisition preserve PLE history and chunk frontier; all stage checkpoint parts name the same complete prefix |
| Two live groups, one cancellation during bulk | Cancelled slot retires once after its consumers drain; other group completes; no reuse of deferred replay or request identity |
| One rejected MTP proposal | Existing accepted-prefix commit restores rejected recurrent/index state before draft advances; no rejected token is published |
| Unsupported borrow/helper/geometry and SYCL | Explicit refusal before entering retained bulk work; feature-off control remains available |

`STRATA_PREFILL_RETAINED_TRACE=1` prints stage, next-layer cursor, local-idle
witness, in-flight groups and running stages at each callback. The repeated final
cursor on ancestor reacquisition is an epilogue, not a repeated layer. This
observer adds stderr work at each boundary; use it only for the short path proof,
then turn it off for speed. State hashes and residual dumps also perturb timing.
Token equality is diagnostic; it is not a universal quality acceptance threshold.

Coordinator build dependency closure: `include/strata/prefill/prefill.hpp` affects
all Prefill consumers (generation, serial judge, allocation helpers and associated
tests); rebuild CUDA/HIP generation and Prefill objects and the SYCL counterparts.
No device kernel source changed. Build through the shared `window.py run --class
build` gate, then use the normal measured host launcher with <=1800 seconds,
8 GiB physical reserve and unchanged thermal/resource stops.

Exact initial runtime command, after the coordinator creates an isolated profile
with the pinned candidate binary and flag in its `env` (the file does not yet exist):

```sh
python3 /home/joshkirklin/ai-optimise/tools/host-inference/measure.py --profile /home/joshkirklin/strata-workspace/scratch/generation-campaign-20261014/retained-on.json --cases chunk --repeat 1 --run-id retained-layer-chunk-on --out /home/joshkirklin/strata-workspace/scratch/generation-campaign-20261014/retained-layer-chunk-on.json
```

Use the same invocation with `retained-off.json`, `retained-layer-chunk-off` and
its separate output for the matched flag-off control. The existing `chunk` case
checks solo/tail behavior only; it does not prove overlap. The coordinator must
use its bounded concurrent generation harness for the two-request scenario above,
with both response clocks and actual interior witness exposure. Do not substitute
the existing 24-arrival responsiveness sweep for this small functional check.

Unresolved: compile/link validation, actual graph/scratch interactions, sampled
state comparison, cancellation runtime behavior and speed. Local stream joins,
serial bulk handoffs, synchronous commit/draft and global admission drains may
consume the modeled benefit. Measure whole-stream gaps, queue-inclusive TTFT,
complete responses per second, bulk work and allocation receipts; no improvement
or release readiness is asserted here.

## Frozen campaign reconciliation

The coordinator's `evidence/generation-campaign-20261014/plan.json`, frozen at
2026-10-14T15:13:31.510823+00:00, supplies the prospective acceptance threshold:
use its active-decode fixture, then cold-prefill after first visible content;
require a >=100 ms whole-stream largest-gap reduction in both comparison orders,
without a reproducible complete-cohort regression or reduced residency. These
are the coordinator's fixed criteria, not a threshold selected after seeing data.
The hand-derived event witness above passes only the idealized reachability
prerequisite. No real candidate acceptance condition has been exercised here.
The `stage` control uses the same retained owner machinery and differs only in
interior callback frequency, so it isolates the scheduling cut more closely than
comparing a scoring lane with ordinary generation admission.

## Retained cancellation repair (source only)

Repair baseline: `85d14713c660c5dc15cf8017a5258c96512aaaba`. The parent's
`/home/joshkirklin/strata-workspace/scratch/generation-campaign-20261014/results/retained-disconnect-e1/result.json`
is negative evidence and must remain unchanged, together with
`/home/joshkirklin/.local/state/strata-host/gen-retained-disconnect-e1/engine.log`
and `frontend.log`. The disconnect followed an interior witness with
`stage=0 next_layer=1 local_idle=1 inflight_groups=1 other_running_stages=1`.
Native incomplete-prefill cancellation and frontend attribution passed, but the
successor failed with `prefill: owner rebind while scratch is in flight or poisoned`
and the peer failed when the engine exited with status 1. Owned teardown passed;
there was no thermal or resource stop. This is a correctness failure, not an
accepted cancellation result or a performance measurement.

The cause is the distinction between stopping a request and failing its arena.
The ordinary chunk stop marks its lease complete after a canary check; the
retained boundary previously returned false through the generic poisoning path.
Clearing poison afterward would also hide failed producers and incomplete joins.
The correction instead records intentional stop intent separately from the
validated cancellation result. It changes these files:

- `src/prefill/prefill.cpp`: checked retained cancellation cleanup and descendant
  propagation, shared root stop predicate, exception-safe pipeline drain.
- `src/program/generate.cpp`: preserve the bulk outcome across the peer drain;
  retained failures other than validated cancellation set `batch_fatal`, even
  when STOP races the error.
- `tests/program/prefill_interleave_test.cpp`: extend the existing executable
  `read_part` seam with injected terminal outcomes and a peer drain that overwrites
  the shared error string. There are no source-text assertions or CUDA stubs.
- This experiment document: causal reasoning and the parent's validation plan.

The lifetime argument is based on source inspection, not execution:

1. Each boundary first joins the local bulk streams, then lets the existing
   callback reacquire its device from decode. The root stop predicate applies
   to descendants too. A callback or device-restoration failure never sets stop
   intent. Entry cancellation does not validate an unstarted canary or grouping
   counter.
2. An early return destroys chunk-local issuer/stager guards before the run
   lease. The retained PLE guard consumes the pending future, while its capture
   storage is still alive; false results and exceptions poison the arena. The
   all-resident admission excludes helper and CPU expert work, and any resident
   streaming issuer has no missing-expert jobs. The lease still drains helper
   streams defensively, followed by compute, expert-copy and KV-copy streams.
3. Only after those joins does the lease validate the started canary, initialized
   native grouping error word and native profile events. It records cancellation
   only with no poison and no active exception. A failed stream join terminates
   rather than releasing an owner of unknown completion. Nothing clears poison.
4. Retained child calls are synchronous and keep parent handoff buffers alive.
   A validated child cancellation makes each ancestor reacquire its own device
   before its lease drains; callback failure poisons that ancestor. Public
   `run()` retains its lease while recursively draining futures, including after
   body/future exceptions. In this tree the recursive method is named
   `drain_pipeline`, not `drain_next`. A failed join takes precedence over a
   cancellation result. Successful completion is marked after the KV join.
5. The serving caller still drains all verifier groups before returning from
   `read_part`. A partial publication or failed peer drain remains fatal. Prompt
   failure leaves `pp_exact=false`; `live_ok` was invalidated before the read,
   and outer checkpoint advancement is after the success branch. The final-stage
   chunk callback only assembles a periodic checkpoint from a complete set of
   stage parts; `part_at` is cleared at the next request. Thus reusable scratch
   does not make partially updated GDN/PLE/KV or a partial stage checkpoint into
   a valid conversation. Previously complete checkpoints retain their existing
   rules; a successor must restore or reset session state normally.

The successful arithmetic, callbacks, 2048-row partition, final tail and device
allocations are unchanged by source inspection. Additional joins/checks and
ancestor reacquisition occur on the cancellation/error path. The flag-off
ordinary chunk-stop path is retained; common error unwinding additionally joins
KV and drains descendants after exceptions. No exported signature or header
changed. No compilation, tests, lint, formatter, GPU work, services, speed,
sampling or promotion checks were run for this correction.

The smallest CPU regression is the existing `prefill_interleave_test` target.
Its new cases require one peer drain after cancellation, cleanup/publication
failure and normal completion, fatal peer-drain failure, and preserved original
bulk errors despite error-string aliasing. Reverting the caller guard makes the
cleanup/publication cases nonfatal; reverting the saved error makes the cancel
case fail. It does not exercise `RunLease`, CUDA streams, PLE joins, descendant
device reacquisition or state reuse. Those require the real engine fixture.

Parent acceptance remains one rerun of the same real HTTP disconnect scenario,
after rebuilding the CUDA/HIP dependency closure for Prefill and generation and
the CPU regression target. SYCL has no changed implementation and still refuses
retained mode; retain that build/refusal check in the parent's backend closure.
Do not rerun the original failing candidate. Use a new binary receipt, profile,
frozen plan, run ID and output directory, preserving the original fixture,
model, sampling, batch/residency, acceptance assertions and resource stops.

The existing external driver hard-codes both the old source and old binary
reference. The parent must prepare
`retained_prefill_disconnect_repair.py` as a copy with `SOURCE` set to this repair
commit and `REFERENCE` set to a newly frozen reference carrying the rebuilt
binary/profile; keep its original behavioral checks intact. Include the original
reference and disconnect evidence hashes in the new provenance. Prepare
`inputs/retained-cancel-repair-e1/plan.json` with run ID
`gen-retained-disconnect-repair-e1` and a new result directory. Neither the copied
driver nor that plan was created by this source-only worker. Once prepared, the
precise parent invocation is:

```sh
python3 /home/joshkirklin/ai-optimise/tools/host-inference/window.py run --class measure --ticket retained-cancel-repair-e1 --purpose retained-cancel-correctness --max-s 600 --wait-s 1800 -- env GENERATION_MEASURE_WINDOW=measure python3 -B /home/joshkirklin/strata-workspace/scratch/generation-campaign-20261014/retained_prefill_disconnect_repair.py /home/joshkirklin/strata-workspace/scratch/generation-campaign-20261014/inputs/retained-cancel-repair-e1/plan.json
```

Require the interior overlapping-work witness, attributable incomplete-prefill
cancel, peer completion, successful successor without poisoned-owner reuse,
and clean owned teardown. Keep the 8 GiB physical reserve and existing thermal
and resource stops. Zero cached successor input alone is not an internal-state
fingerprint proof. Stop on the first functional failure and retain the evidence.
Remaining risks are uncompiled CUDA/HIP cleanup, real doorbell/stream progress,
PLE failure handling, later-stage cancellation and checkpoint/state equivalence;
the single stage-0 disconnect rerun cannot qualify all of those paths.
