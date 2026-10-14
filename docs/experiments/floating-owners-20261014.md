# Floating two owners over four states

Source-only experiment based on 335b954daa473fead32e17acd9e7e898944115f5.
No build, test, lint, formatter, model execution or GPU operation was performed.
No speed, numerical equivalence or runtime correctness result is claimed.

`--batch-mtp-floating` selects ready stationary seats in the existing batch-MTP
pump. It requires `--batch 4 --batch-groups 2 --batch-mtp-pipeline`, including its
existing distinct-device/MTP restrictions, and refuses a reduced effective seat
count after allocation. Without the new flag the original selection, row order,
launch, acceptance, commit and draft calls remain on their original branches.
The added default-false field/help/parser and stack metadata do not establish
binary identity; the intended off invariant is unchanged inference operations.

Selection uses oldest last-service tick, breaking ties by seat number. A packet
has at most two seats; when both owners are idle, two ready seats are split into
two singleton packets. Four ready seats form two pairs. One ready seat launches
immediately. The selected pair is sorted before adding current/proposal rows.
The existing bounded-round owner opportunity limit remains in force; this does
not modify prefill continuations or their arithmetic partition.

A seat appears in at most one inflight PGroup. PGroup remains inflight through
all target stages, recursive accepted-prefix commit, emissions, finish fencing,
and synchronous draft advance/propose. Only then may either owner borrow that
seat. No SessionOwner, history, sampler, request ID or draft state is moved.
Cancellation while leased follows the original keep/finish path; unleased
stopped seats pass ResidentSlots::finish before reuse. When retirement is
forbidden, an unleased stopped seat conservatively delays new floating launches.
Failure returns without clearing the packet lease. Existing shutdown/drain owns
cleanup; this experiment adds no recovery shortcut.

Fixed handoff bases are still 0 and 4, each with at most four rows. Each owner's
canonical pair/singleton schedule has 6*4 + 4*2 = 32 keys, including depth-one
boundary rows. Verifier::reserve_batch_key still refuses overflow and never
evicts live graphs. Both owners already bind all four states. Existing primary
and peer graph allowances, replay/canary pricing, host staging and per-device
admission remain unchanged. Increased reachable graph occupancy is not zero
cost: actual retained storage must remain inside the admitted allowance/MPS
ceiling; any refusal is retained, not worked around by reducing experts.

## Coordinator verification plan

Build dependency closure: only src/program/generate.cpp changes. Rebuild/relink
`strata` for CUDA and HIP (including gfx906 if qualified); the optional
strata-resident-test, strata-judge-test and strata-residency-test targets compile
that same translation unit. Its linked closure remains strata_resident,
strata_engine, strata_prefill, strata_spec and the backend runtime. No header,
kernel or library implementation changes. SYCL resolves its separate
sycl/src/program/generate.cpp, so this flag is not implemented there. Builds are
reserved to the coordinator through shared window.py build, not run here.

The requested exact engine argument delta is `--batch-mtp-floating` added to the
campaign's pinned admitted profile engine_args plus `--serve`; control omits
that one argument. Preserve all other arguments, including `--prefill 2048`,
`--max-context 262144`, `--layer-split 20,39`, four slots, both owners, MTP/min-p,
model, expert profile and graph allowance 268435456. Use the coordinator-built
binary rather than the profile's old binary. Run only through the coordinator's
measure/host monitor with a finite <=1800-second campaign limit and 8-GiB host
reserve; the daily profile's timeout=0 must not be copied as the experiment
limit. No production profile edit is requested.

Use existing native BGEN seat assignment for causal functional proof. Use
complete frontend response cohorts separately for user-visible throughput.
Short retained prompt fixtures avoid measuring fresh admission in the tail.

| Scenario | Required assertion |
| --- | --- |
| Four requests; long survivors in seats 0/1, early finishes in 2/3 | After old packets drain, both owners lease 0/1 separately without state copy or re-admission; baseline retains static owner affinity. |
| Matched ordering; survivors 0/2 | Negative control: both policies can use both owners; report any packet-size or latency cost. |
| Four symmetric short requests, then one remaining request | No starvation, duplicate lease or wait-to-fill; unchanged four-state capacity. |
| Cancel during each target stage and after target completion before commit, then immediate same-seat reuse | No reuse until all target/worker/draft consumers drain; no old request history/sampler/state crosses identity. A live peer completes. |
| One accepted proposal and one rejected proposal; short seeded non-greedy continuation | Keep counts, accepted recurrence/indexer/PLE/KV and draft extents agree with the existing commit contract; rejected proposal does not survive rollback. Greedy equality alone is insufficient. |
| Context/output boundary depth-one rows; canonical singleton/pair combinations | No key overflow above 32 per owner and no overlapping handoff row range; four full-capacity reservations and all 24,576 experts retained. |
| Ordinary bounded decode round and stage-restricted judge coexistence, including cancellation | Round terminates, no lease transfers before retirement permission, no prefill continuation change. |
| Flag omitted; same inputs and settings | Existing baseline path/committed-state diagnostics unchanged; token equality is diagnostic, not a universal acceptance gate. |

For short functional runs only, `STRATA_STATE_HASH=1` also emits
STATE_FLOAT_LEASE (tick, owner, seat, position) and STATE_FLOAT_RELEASE after
commit/draft drain. Match these with existing request/state/proposal records;
an owner may have two lease lines per packet, one release. These observations
add synchronous stderr work, and existing hashing synchronizes/copies state;
never use that run for speed. Lease records are intent immediately before launch;
a launch failure is not successful execution evidence. No new device memory or
retained event buffer is allocated. Set no state-hash variable for timing runs.

Collect complete response time, committed throughput, per-request gaps, and
ready-wait intervals with the same observer on/off. No STRATA_VERIFY_PROFILE:
it changes overlap and its concurrent storage is unpriced. A change in owner
assignment alone proves neither a bottleneck nor a gain. Refuse on identity,
state, capacity, expert-residency, graph, resource or safety failure. Smaller
packets can change numerical dispatch and lose expert amortization; these remain
unresolved runtime risks. No promotion or publication is part of this delivery.
