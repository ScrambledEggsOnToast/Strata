# Stationary ownership and direct-to-slot prefill

Separate evaluation against source `335b954daa473fead32e17acd9e7e898944115f5`.
No stationary-ownership implementation or runtime result is claimed. This is
independent of the [retained-layer experiment](retained-prefill-model.md).

## Falsified shortcut, viable larger proposal

Changing only `Prefill::rebind_session` cannot implement direct-to-slot generation.
The concrete counterexample is a fresh request whose prompt is written into slot
S, followed by the unchanged first-token path and `ResidentSlots::admit`:

1. Prefill writes S's recurrent/KV/PLE state, but the solo verifier remains bound
   to the working session. Its first-token calculation consumes a different state.
2. `on_chunk` calls `Prefill::draft_kv` or `MtpDrafter::prefill` against the working
   draft `st_`, not S's private draft state. `slot_state` exposes a const view,
   not a writable prefill lease.
3. Admission still calls `copy_to(S)` from the working main state and
   `save_slot(S)` from working draft state. It overwrites the direct result before
   publishing active metadata and installing both verifier-group contexts.

This falsifies the pointer-only shortcut, not stationary ownership's performance
premise. Copy costs may be material or negligible; no runtime evidence is available
in this wave to reject that premise. A complete migration remains a separate,
viable implementation project rather than a safe adjunct to layer scheduling.

## Complete binding obligation

| Owner / source seam | Required stationary transition |
| --- | --- |
| `ResidentSlots::begin/admit` | Reserve an available destination and request generation before mutation; retain scope, seed, history and failed-state poisoning until publication |
| Every stage's `Prefill` | Validate geometry/device/stream, rebind all stages and device RoPE tables, retain one arena; restore old bindings on every exit |
| Solo target verifier | Bind the selected session for prompt tail and first-token verification, with captured graph address lifetime and canaries; no API currently performs an arbitrary session rebind |
| Both batch verifier owners | Keep stable slot addresses and install request-specific sampling/history/steering in both chains; never rebind a live or deferred owner |
| MTP working prefill | Add a writable slot-prefill operation that binds KV, ring, pooled/indexer metadata, cell extent and operation canary together; no const cast or state-struct swap |
| MTP captured slot graphs | Retain their allocated addresses and per-slot cells/proposal metadata; establish initial proposal only after the authoritative first token exists |
| Admission publication | Replace main/draft copy with validated in-place seal only for migrated requests; active becomes true after every consumer is bound |
| Resume, cache, park and cancellation | Chosen checkpoint restores into the same destination including draft repair; partial writes poison it, and cancellation drains borrowers before reservation release |

Relevant concrete source: `src/program/resident_slots.cpp` (`idle`, `slot_idle`,
`admit`, `copy_to`, `resume`, `committed`), `src/core/mtp.cpp` (`save_slot`,
`restore_slot`, `advance_slot`), `src/core/verify.cpp` (`init_slots`,
`set_slot_context`, `commit_slot_prefixes`), and `src/prefill/prefill.cpp`
(`validate_rebind`, `rebind_session`, `draft_kv`). `SessionOwner` remains nonmovable;
logical ownership cannot move captured allocations by swapping owners.

## Causal work bound and small discriminator

For the actually live prefix length L, admission currently transfers approximately
`2 * sum(stage_checkpoint_and_KV_bytes(L)) + 2 * draft_KV_bytes(L)` through host
buffers, excluding allocator and synchronization overhead. Reservation capacity
is not the transferred extent. Ordinary resident token commit does not do this
working-to-slot transfer, so uninterrupted decode cannot gain from removing it.

Before funding the full binding migration, use the existing
`STRATA_ADMISSION_TIMING=1` seams on one fresh admission and one retained resume,
then one observer-off matched control. Count the actual checkpoint/main KV/draft
payloads. If removable critical-path time is fraction f, its ideal response
speedup is at most `1/(1-f)`; this is an algebraic bound, not a measured fraction
or an operator-approved threshold. Price exact same-device copies as a cheaper
alternative if they capture the material part of that bound.

A subsequent implementation must separately demonstrate fresh first token,
sampled rejection, retained checkpoint resume, cancellation before publication,
and reuse of a retired slot. Assert main AND draft ownership at each transition.
Do not delete or claim savings for the working allocation until solo generation,
prompt tails, scoring, cache restore and every other borrower have migrated.
Four maximum-context slots, full expert residency and host reserve remain fixed.
