# Third rolling packet with four request states

Separate source/model evaluation against 335b954d plus the floating-owner change.
No implementation, build, test, allocation measurement or GPU run is claimed.
This rejects adding a third PGroup to the existing owner machinery; it does not
reject the possibility of a properly admitted third independent owner.

The source supplies a concrete falsifying ownership trace for a scheduler-only
third packet: packet B owns group 1's deferred replay at stage 0 and advances to
stage 1, while packet C calls group_verifier(0, 2). The selector returns
stage_peer for every nonzero group, so C aliases B's uncommitted replay, metadata,
worker dispatch and graph owner. Poll completion does not release that lease;
commit_slot_prefixes still needs B's captured intermediates. Treating group 2 as
an independent descriptor therefore violates accepted-prefix ownership before
any performance experiment. No safe third packet exists merely by changing the
PGroup vector size or --batch-groups admission guard.

Other independently necessary migrations are source-visible:

- generate.cpp currently admits exactly two pipeline groups and chains primary
  and peer verifiers. A third verifier chain needs distinct canaries and worker
  completion/dispatch ownership on every stage, not another alias of split_drive_b.
- ResidentSlots::idle and slot_idle fence primary and batch_peer; slot_idle scans
  exactly two completion entries per stage. Resource initialization/release and
  request history/sampling/steering binding cover these two chains. A third owner
  must participate in every fence and borrower-lifetime operation before use.
- Existing handoff base is owner*min(2*GS,4). Three four-row packets reach row 12,
  exceeding kVerifyMaxT=8. A valid layout with four seats is 2+1+1 seats, bases
  0/4/6 and row capacities 4/2/2. Owner capacities must remain fixed for captured
  address/key lifetime. This changes kernel packet shapes and useful work per
  circulating packet; it cannot inherit a fixed-shape speed prediction.
- Existing admission prices one additional replay/control/canary envelope and
  one additional graph allowance per stage. A third independent chain must add
  its own device and host demand before expert residency is assigned. Four
  maximum-context seat arenas and their private drafts cannot be reduced to
  fund this demand. The working state remains required too.

ResidencyAstra's retained receipt derives a conservative third-owner increment
of 67,440,368 + 268,435,456 = 335,875,824 bytes per device, or 1,007,627,472 bytes
across three devices, before new handoff/host/worker bookkeeping. These are old
receipt-derived bounds, not current measured free capacity. A 2/1/1 design may
price smaller singleton scratch, but the complete current per-device allowance
must be recomputed, not divided by a convenient factor. No evidence in this
source-only wave establishes that the unchanged caps and all 24,576 populated
experts leave this room. Missing headroom evidence is unresolved, not a proof
that three owners can never fit.

The bounded closed-pipeline model is

    packet_rate <= min(1 / max(stage_time), owners / (sum(stage_time) + overhead))

For equal stages, zero overhead and fixed packet shape, moving two owners to
three has an ideal 1.5x ceiling. If the slowest stage already saturates, adding
an owner cannot exceed its service bound. With 2+1+1 instead of 2+2 seats,
service time and committed tokens per packet change, so even that ceiling is
not a prediction for this candidate. Shared synchronous drafting remains a
separate circulation constraint.

Disposition: do not submit a third-owner runtime flag from this wave. The
scheduler-only proposal is rejected by the aliasing/row-range counterexample;
a full third-owner implementation remains conditional on explicit independent
ownership, current per-device admission with four long states and full expert
residency, and favorable measured service demand. Tiered state, immutable RoPE
sharing and prefill continuation changes were not introduced as hidden funding.
A future bounded capacity check needs no 256K generation: prove four long-state
reservations fit and any optional third owner drains before relinquishing its
budget, then separately exercise short actual packets, cancellation and reuse.
