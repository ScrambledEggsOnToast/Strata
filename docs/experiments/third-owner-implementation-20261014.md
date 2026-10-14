# Independent third execution owner

Source-only implementation based on 6a0774a1, in the separate
`generation-third-owner-20261014` checkout. The two-owner checkout was not changed.
No build, tests, formatter, GPU operation, download or service work was performed.
The prior third-packet note rejected an aliasing shortcut; this change implements
an independent chain. It does not claim a measured third-owner gain.

## Current admission calculation

Inputs are the current source allocator formulas and the retained
`generation-campaign-20261014/runs/gen-groups-speed-off-e1/engine.log`, not the
older research receipt. Each additional owner uses the same conservatively
four-row arena even when its packet holds only a singleton:

| Increment | Bytes |
| --- | ---: |
| Replay + four-seat controls + independent device canary, per device | 67,440,368 |
| Additional graph allowance, per device | 268,435,456 |
| Device slot-offset tables, CUDA0 / CUDA1 / CUDA2 | 81,920 / 77,824 / 36,864 |
| Device increment, CUDA0 / CUDA1 / CUDA2 | 335,957,744 / 335,953,648 / 335,912,688 |
| Device increment across three devices | 1,007,824,080 |
| Three additional mapped/runtime host envelopes, histories, controls, canaries and diagnostic buffers | 303,174,720 |
| Six additional worker ledgers/operation/GraphExpertWorker owners | 20,736 |

Host metadata is additionally charged as
`sizeof(SplitDrive) + 3*(sizeof(Verifier) + sizeof(unique_ptr<Verifier>))`.
The compiled ABI value remains to be emitted by admission; it was not guessed
or obtained by compiling in this wave. The conservative host runtime envelopes
also cover bounded scheduler and allocator metadata. No handoff payload is
added: the existing eight rows are partitioned, not enlarged.

The retained admission sums are CUDA0 32,232,651,084, CUDA1 32,234,522,452 and
CUDA2 21,552,430,420 bytes before headroom. Existing auto cache budgets already
include slack above actual full-pair allocations. Applying the extra device
charge to the source budget formula, with unchanged ceiling, 734,003,200 bytes of
headroom and 2,662,400 maximum blob size, originally gave CUDA1 22,760,857,600
and CUDA2 10,607,001,600 bytes for experts before the slot-offset correction.
The additional offset charge is smaller than one maximum blob on each device,
so conservative corrected lower bounds subtract one blob from those budgets.
Their full caches print 18.69 and 9.69 GiB; even rounding upward by 0.005 GiB
bounds them below 20,073,603,400 and 10,409,926,984 bytes respectively. The last
device retains at least 194 MB against that rounded upper bound, before cache
allocation granularity. This is a source/receipt calculation, not a new physical
memory observation.

At actual startup the implementation computes the exact 256-aligned all-pair
bytes from the loaded native layout, requires the newly admitted budget to
cover them, charges the existing 2-MiB cache rounding, then checks actual cache
bytes equal those exact extents and every expert is present on its owning
stage. The existing unknown-class, per-device, host/cgroup and graph allowance
admission gates remain authoritative. No expert removal funds this experiment.
Review found that existing prompt initialization could shrink chunks or change
residency after the early check. Third-owner startup now refuses both chunk
fallbacks and rechecks all three actual prompt chunks plus full residency after
prompt setup; only that final gate emits the capacity witness.
A fresh measurement may still refuse; no reserve/cap relaxation is authorized.

## Implementation and isolation

`--batch-mtp-third` requires the existing floating four-state/two-group pipeline
configuration, three explicit distinct-device stages, `--spec 4`, `--prefill
2048`, and static full native local expert residency. It rejects judge,
multiplex, pipeline-windows, split fallback, adaptive and helper compositions.
The original four maximum-context session/draft states and working owners stay
allocated; failure to retain all four still refuses startup.

The third chain is allocated only with this flag. Each stage receives a new
Verifier with its own arena, graph maps, stream set, canary, slot histories,
commit controls and replay buffers. It has its own SplitDrive and worker lanes
3/4/5, mapping to completion/operation/worker entries 6..11. The original lanes
0/1/2 and their six entries remain intact. The split adapter validates lane
plus chain offset. Global worker scheduling and synchronous draft scratch are
still shared as in the control; their consumers are drained before reuse.
Nine initialized verifiers fit the existing 16-owner watchdog release registry.

Packets use fixed seat capacities 2/1/1 and fixed physical bases 0/4/6. A
singleton contributes one current row and at most one proposal; a pair contributes
at most four rows. The graph-key bound is 32 for owner 0 and 8 each for owners
1/2 (the existing 32-key refusal limit and full graph allowance remain charged).
Stationary seats are excluded from selection while any packet leases them.
Target sampling, acceptance, recursive prefix commit and private draft advance
are unchanged; a packet releases only after those steps finish.

ResidentSlots now includes the third owner in idle/slot-consumer checks, initial
and restored request-context binding, release/reinitialization, partial-failure
cleanup and destruction. A new request cannot replace history or identity until
all three chains and their worker consumers drain. The third chain receives
per-request sampling/history and bias clearing alongside the control chains;
per-slot identities, steering and resolved seeds bind independently through the
existing GPU canary path. No prefill continuation or kernel implementation changes.

With the flag absent, no third verifier, dispatcher, worker ledger, graph,
device/host allowance or scheduler packet is allocated. The original owner
selection/capacities, row bases and inference calls remain. Optional pointer
fields and stack descriptors exist in the source layout; off equivalence means
unchanged inference operations and payload allocations, not binary identity.

## Coordinator build and run

Changed engine files: `src/program/generate.cpp`,
`src/program/resident_slots.cpp`, `include/strata/program/resident_slots.hpp`.
Rebuild `strata_resident` and `strata`, plus affected optional resident/model,
judge-allocation and residency executables. Direct header dependents include
`tests/core/resident_slots_test.cpp` and `tests/core/resident_model_test.cpp`.
The linked closure remains strata_engine/strata_prefill/strata_spec and backend
runtime. CUDA and HIP share these changed sources; builds belong to the
coordinator's queued window.py build flow. SYCL has its separate generate.cpp
and does not build this resident module; the new flag is not ported there.
No backend is claimed built or validated by this source delivery.

Use the exact retained speed-off-e1 engine argument list with the newly built
binary, preserving the model, steering, three device caps, 262144 context,
2048 prefill, static full expert set, MTP/min-p and other arithmetic settings.
The exact control/candidate argument additions are:

    control:   --batch-mtp-floating
    candidate: --batch-mtp-floating --batch-mtp-third

Keep `--batch 4 --batch-groups 2 --batch-mtp --batch-mtp-pipeline --spec 4
--prefill 2048 --layer-split 20,39 --split-device 1,2`; batch-groups remains the
four-state control setting, while the new flag selects three execution owners.
Run only through the coordinator's measure/host monitor with finite <=1800 s,
8-GiB physical reserve and unchanged thermal/resource stops. Do not edit the
daily profile. For functional observations only add `STRATA_STATE_HASH=1`;
for idle resource cycling also set `STRATA_BATCH_DIAGNOSTICS=1`.

## Functional witnesses and assertions

The opt-in observer emits STATE_THIRD_CAPACITY with actual cache bytes and
unchanged slot/context/chunk values, STATE_THIRD_OWNER with distinct owner
addresses and worker lanes, STATE_THIRD_LAUNCH only after successful launch,
and STATE_THIRD_STAGE_DONE only after successful poll. Existing STATE_FLOAT_LEASE
and STATE_FLOAT_RELEASE bracket seat leases through commit/draft completion;
existing request IDs, canaries, state fingerprints and proposal receipts remain.
These synchronous stderr and state-hash operations perturb host timing and may
synchronize devices; omit them entirely from throughput measurements. Failed
launch intentions alone are not execution evidence.

| Bounded scenario | Required evidence |
| --- | --- |
| Four short retained prompts, enough output for several pipeline rotations | Owner 2 successfully launches and completes on all three stages before releasing; capture a period with all three packets live. Distinct owner addresses, row bases 0/4/6, maximum row counts 4/2/2, no seat in two live leases. Absence of this exposure is inconclusive. |
| Four full-capacity allocations with short actual requests | Exact admission/cache extents, four 262144 states and four private drafts retained, 24,576 local experts before/after, unchanged caps and host reserve. No long generation is needed. |
| Same-group survivors 0/1, matched opposite-group survivors 0/2, plus symmetric four-request control | All remaining requests progress, no wait-to-fill; compare against two-owner floating, not static grouping. Two survivors cannot occupy three owners simultaneously. |
| Cancel a request witnessed on owner 2 at stage 0, 1, 2, and after final completion before release | Cancellation does not permit another lease or same-seat request identity before recursive commit and draft drain. Live peers finish; immediate same-seat reuse receives a new canary identity and isolated history/steering/seed. If timing misses the intended phase, do not mark it covered. |
| Accepted proposal, rejected proposal, output-limit singleton row, seeded non-greedy continuation | Actual acceptance/keep counts and rollback extents; matching committed recurrence/PLE/indexer/KV/draft invariants. Sampling remains position/seed based. Greedy equality is diagnostic, not proof of the sampled law. |
| Idle BRESOURCES release/init, then same-seat request | All three graph/resource chains release before borrowed state is rebound; successful post-reinit canaries/owner-2 execution. Repeated unlimited cycles are not required. |
| Partial init/capture/launch failure and shutdown, using existing authorized fault seams where available | No early resource reclamation or stale-worker publication; all registered verifiers drain. No fault-injection coverage is claimed here. |
| Flag omitted in the same new binary | Original floating path, two owners, original admission class values and packet behavior. No third-owner observer lines or payload allocations. |

After functional obligations pass, compare complete response cohorts with all
observers off, preferably paired in reverse order. Record committed throughput,
response completion times, request gaps, admissions and memory, not GPU usage
alone. Small singleton packets can lose expert amortization; shared draft/commit
service may dominate. Changed packet shapes may change floating-point dispatch.
Current model arithmetic and target acceptance code are preserved, but no
numerical, isolation, quality or performance pass is claimed until exercised.
Three-owner judge/helper/SYCL compositions and broader geometry are excluded.
