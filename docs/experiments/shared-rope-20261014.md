# Shared immutable RoPE experiment, 2026-10-14

Source-only candidate on 335b954daa473fead32e17acd9e7e898944115f5. No build, test,
formatter, GPU operation, service operation or measurement was performed in this wave.
Read against ResidencyAstra.md and Astra-Contrarian.md from the generation research.

Set exactly `STRATA_SHARED_ROPE=1` before starting the engine. Unset or `0` retains
legacy arena pricing/carving, table construction and release. This switch does not
turn on `STRATA_ROPE_TABLE`, change native/analytic rotation, or change arithmetic.
The existing float64 host builder produces the same table contents. Four full-context
slots, working state, draft state, expert residency and prefill2048 remain required.
No scheduling, session binding, model, routing or sampling code was changed.

## Ownership and pricing

The table lives outside QSA/session arenas. A fixed 64-device registry admits one
exact `(device, max_cells, n_rot, all RopeScaling fields)` key per device. A second
incompatible key refuses initialization instead of replacing captured addresses.
Different judge context is refused early by generate when the switch is enabled.
This deliberately bounded candidate does not implement a multi-model table cache.

Published tables remain allocated until process exit, including after all sessions
are released or an initialization fails later. No static destructor calls CUDA.
This retention is intentional and charged, not a session leak: a graph may retain a
raw table pointer after its supplying session disappears. A refcount on sessions
alone would be insufficient. The table never mutates, grows, or gets evicted; device
reset/reinitialization in this process is unsupported. Other mutable session storage
still requires its existing borrower-drain contract; retaining RoPE does not make
replaying a whole model graph after session retirement valid.

Initialization serializes table publication, uploads both arrays before publishing,
and frees an unpublished allocation on upload failure. After publication, session
unwind neither frees the table nor clears its native registry entry. Borrowed MTP
and layer views keep using the existing addresses.

For context C and rotary width R the separate device payload is `C*(R/2)*4*2`.
Native admission adds this once per stage device and charges the fixed host registry
plus mutex once. Each working/resident arena omits its primary table. Host construction
peak keeps the existing conservative serial table/page-table payload; allocator and
driver metadata remain covered by existing allowances. Preallocation split-skip and
auto-split capacity calculations also subtract the separate table. SessionOwner's
reported device bytes continue to mean its own arena, not process-global allocations.
External callers of session pricing must add the documented shared charge themselves.
The independent Python planner remains conservative; it does not claim this saving.

At C=262144, R=64: one table is 67,108,864 bytes (64 MiB). Working plus four slots
formerly paid five tables per stage; the candidate pays one, saving 268,435,456
bytes per stage, 805,306,368 bytes (768 MiB) over three distinct devices. This is
allocation arithmetic, not measured memory, speed, or permission to evict experts.
For arbitrary C, use the allocator's aligned arena differences instead of assuming
all alignment deltas vanish. Draft tables already borrowed and earn no extra credit.

## Build closure and finite verification request

Changed source: `src/core/layer.cpp`, `src/core/session.cpp`, `src/program/generate.cpp`,
`include/strata/core/layer.hpp`, `include/strata/core/session.hpp`, and
`tests/core/session_owner_test.cpp`. Headers require rebuilding all dependent engine,
prefill, program and test translation units; do not reuse objects compiled against
old headers. Rebuild `strata_engine`, `strata`, and `session_owner_test`
through the normal CUDA and HIP configurations; check the repository's SYCL closure
where those shared files are built. The allocation-injection runtime proof is Linux
CUDA only, using the existing linker-wrap target. No backend build is claimed here.

Coordinator owns builds through shared window.py build and GPU runs through the
measure window and ordinary host monitor, with explicit devices, <=1800 seconds,
8-GiB host reserve and unchanged thermal/resource stops. The exact new diagnostic
payload, from the chosen build directory, is:

```
env STRATA_SHARED_ROPE=1 STRATA_ROPE_TABLE=1 ./session_owner_test --shared-rope-gpu
```

This is an explicit opt-in payload for that monitored window, not an instruction to
run outside it. It does not require the retired protected-supervisor receipt. The
existing `--gpu` mode and its guards remain unchanged and require shared RoPE off.
Default host-only pricing can run with either switch value. One diagnostic invocation
is sufficient for these assertion mappings; do not repeat an unchanged failure:

| Finite scenario | Assertion |
| --- | --- |
| Fail each of the two initial table uploads; fail separate table malloc | Failed unpublished storage and session arena/pinned storage fully unwind |
| Construct one working plus four short test sessions | All five share cos/sin addresses; live device bytes equal sum of arena prices plus exactly one table |
| Capture both table reads, then destroy every supplying session | The graph still borrows valid immutable addresses |
| Fail a reused-table session page upload; request incompatible context | No published table replacement, leak of failed owner, or dangling registration |
| Replay once and compare every cosine/sine byte to the unchanged host builder | Captured immutable content survives all preceding retirement/failure paths |
| Destroy graph and its output | Only the explicitly charged process-lifetime table remains |

The diagnostic uses C=65 and a small layer range, not near-maximum-context generation.
Its copy graph proves table lifetime and byte identity, not model-graph correctness
or multi-device isolation. It allocates observer buffers, performs readbacks, and
injects failures: never use its timings for speed claims.

After successful build/diagnostic, request one matched normal four-slot cohort using
the existing finite campaign profile with only this environment toggle changed.
Keep C=262144 allocation, all 24,576 expert pairs populated, identical layer split,
sampling, chunk2048 and graph allowances. Verify the native admission's shared_rope
class is 64 MiB on each device, four slot arenas fit, no expert loan/offload occurs,
and cancellation followed by same-seat reuse and a live peer reconcile request IDs
and counts. Existing state hashes can compare recurrence/KV/draft state after one
rejected-prefix commit and continuation; token agreement alone is not the assertion.
No 256K prompt or throughput claim is needed to verify allocation capacity. Observe
allocation records with STRATA_STATE_HASH only in the functional run; speed comparisons
must omit that observer. Performance, runtime success and cross-device isolation remain
unverified until coordinator execution.
