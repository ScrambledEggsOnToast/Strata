# Context-tier allocation disposition, 2026-10-14

Source/model proof only, evaluated against 335b954daa473fead32e17acd9e7e898944115f5
and the independent shared-RoPE candidate. No compilation, tests, GPU probes, model
runs or services were used. The frozen campaign calls for four maximum-context
requests, full expert residency, unchanged caps/reserve and chunk2048; this worker
may not edit scheduling or session binding.

Disposition: reject the eager smaller-arena substitution as a usable candidate in
this scope. A bounded dynamically backed allocator remains blocked on a concrete
missing ownership/admission implementation, not disproved as an architecture. There
is no context-tier runtime flag or partially wired implementation in this commit.

## Concrete failing paths

1. In `src/program/generate.cpp`, resident allocation calls
   `SessionOwner::init(g, o.max_context, K, ..., &kv_roles.slot_main)` for every slot
   and stage. `qsa_role_placements` freezes each role's `cells` before startup
   admission; `qsa_placement_matches` refuses using that placement with a different
   context. Merely replacing the allocation byte count leaves the existing carve
   larger than its arena and `session_init` refuses it.
2. Re-resolving a genuinely smaller placement and SessionState is still not usable:
   `Verifier::init_slots` in `src/core/verify.cpp` explicitly rejects
   `x->max_cells != ss_->max_cells`. Both existing replay owners bind those slots
   through `ResidentSlots::init_resources` in `src/program/resident_slots.cpp`.
3. Independently, `MtpDrafter::init_slots` in `src/core/mtp.cpp` prices every draft
   arena with `st_.max_cells` and refuses `sessions[i]->max_cells != st_.max_cells`.
   Removing only the verifier guard cannot enable heterogeneous request capacities.
4. Claiming a maximum-context SessionState atop a smaller allocation is unsafe:
   positions are validated against `max_cells`, and device pool, page-table and
   indexer addresses are carved at the resolved geometry. In
   `include/strata/plan/qsa_memory.hpp`, `qsa_device_bytes` prices all resident KV
   pages, page-table entries and pooled indexer rows; `qsa_arena_layout` carves the
   same extents. A short backing cannot satisfy that contract by changing metadata.

These are finite source predicates, not missing benchmark evidence. The existing
host-only `session_owner_test` already asserts that a frozen placement cannot be
repriced at shorter request contexts. No duplicate source-text test was added.

## Capacity proof and rejected accounting shortcuts

Let C be the advertised maximum capacity and let M_i(c) be the actual main plus
private-draft state payload on device i, including alignment, at capacity c. The
current eager resident layouts have positive per-cell cost. With no growth or
rebinding, each of four simultaneously admitted maximum-context requests requires
one private allocation at least M_i(C). Therefore the invariant is

`state_reservation_i >= 4 * M_i(C)`.

Replacing any seat by M_i(c), c<C, violates maximum-capacity support for that seat.
Keeping four maximum-size fallback allocations in addition to short tiers uses
`4*M_i(C) + sum(M_i(c_j))`, so it cannot reclaim memory. Preallocating a maximum
backing pool and carving short live portions preserves the capacity guarantee but
does not lower total device allocation; the unused pool is still reserved. Spending
its bytes on another non-reclaimable replay/graph owner double-counts the same budget.

For the retained 20/39 split, ResidencyAstra's source/receipt model is
`M_seat(c) = 145475088 + 13305*c` bytes across all three devices, up to alignment.
It gives 3,633,301,008 bytes at C=262144 and 581,453,328 bytes at c=32768.
Those counterfactual numbers are not runtime capabilities or new measurements.
Under shared RoPE, subtract the three duplicated per-seat `256*c` table payloads
and separately retain one table per device. This removes immutable duplication;
it does not remove any mutable KV capacity obligation. Four slots still require
four private full-context mutable states. Aggregate savings never establish fit
on a particular device or a third owner's retained graph allowance.

## Why existing alternatives do not remove the blocker

`--kv-grow` is not a qualifying switch for this campaign. Native admission marks
`elastic_kv_growth` unknown because VMM growth/promotion and host metadata peaks are
unqualified. The activation branch in generate also requires `!multi_gpu`, while
this campaign uses three stages. Existing fixed-address VMM is a plausible building
block, not a proved three-device maximum-capacity allocator. A new implementation
would need per-device physical reservation and bounded mapping/growth failures,
without obtaining backing by expert eviction or violating the 8-GiB host reserve.

`--slot-kv-window` is a different existing streaming placement. Its pinned backing
still covers all logical pages (`qsa_pinned_bytes`), and its call/draft live-window
floors are explicit in `qsa_slot_window_error`. It changes transfer demand and
residency layout rather than implementing shorter logical context tiers. Shrinking
or truncating draft history to fit is not permitted. No claim that streaming is
mathematically impossible or necessarily slower is made here.

A reachable dynamic-tier design needs, before execution:

- A maximum-capacity reservation per device that simultaneously backs four long
  requests and full experts under the original caps, including transient growth.
- Stable virtual addresses or drained graph invalidation/rebinding, plus private
  GDN/PLE/indexer/draft state and accepted-prefix rollback through a growth failure.
- Request prompt-plus-output-plus-speculative bounds, checked before mutation; later
  turns must safely grow without silently reducing admitted context or slot count.
- If capacity is lent to optional execution scratch, a certified drain and return
  before long-state growth. Existing two-owner completion/lease accounting cannot
  be assumed to retire an additional borrower.

That migration necessarily touches the forbidden session-binding surface for true
heterogeneous SessionState capacities, or replaces the currently unqualified
multi-device VMM allocation path for fixed logical capacities. It is not justified
by unused-prefix arithmetic alone. No extra hardware or long-context stress test
would resolve these source contracts without the missing implementation.

## Finite assertions and next decision

The eager proposal fails two independent initialization predicates (verifier and
MTP), and the no-growth memory inequality above. This is its falsifying result;
there is no runtime invocation to request for it. Do not load a model just to
rediscover these predicates.

A future dynamically backed candidate must first demonstrate, with its real allocator
pricing and failure transitions, exactly four simultaneous C reservations; growth
of a short seat to C at unchanged caps; one allocation failure leaving the prior
committed state valid; and graph retirement before backing reuse. Those are four
bounded ownership/capacity scenarios, not 256K generation workloads. Only then is a
short mixed-request runtime proof meaningful. Missing implementation and multi-device
growth evidence are explicitly blocked, not successful acceptance. The independently
committed shared-RoPE experiment can proceed without this architecture.
