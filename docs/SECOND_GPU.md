# CUDA1–3 expert experiment

CUDA0 keeps the dense layers, KV/state, MTP and its existing expert cache. Up to
three independent expert caches fill CUDA1, CUDA2 and CUDA3 in that order. Each
expert has one owner: CUDA0, one secondary GPU, or the CPU pool. Each secondary
GPU computes its rows while the CPU pool works; the results return through
private pinned host buffers before CUDA0 continues the layer. The GPU launches
can overlap, but each layer waits for all three.

The shipped profile ranks 8,000 `(layer, expert)` pairs. CUDA0 takes its usual
`--expert-cache auto` allocation. With only CUDA1, it takes the first remaining
pairs. When CUDA2 or CUDA3 is enabled, the remaining profiled pairs are assigned
round robin to the secondary GPUs. The ranked list is then extended with all
other pairs in expert-then-layer order and distributed in the same way. This
fills the additional VRAM and lets all secondary GPUs see frequent experts, but
unranked experts may hardly ever be routed. The startup
log reports slots and GiB per card; each request reports the number of expert
entries actually computed on each secondary GPU. Remote outputs are packed
before returning over PCIe/USB4: the request log compares MiB transferred
with the previous full-row transfer. The input and routing metadata use pinned
host staging. This reduces transfer traffic and launch overhead, but does not
release the expert arena in system RAM or significantly change VRAM use.

The optional `--gpu-placement layer` gives each layer to exactly one secondary
GPU (layer number modulo the number of enabled cards). It fills that card with
the most frequently routed experts of its layers first, then the unranked ones.
This can reduce USB4 device switches and synchronization when several secondary
cards serve the same layer in the default `stripe` mode. All three cards still
keep their requested expert slots; the primary CUDA0 cache and CPU fallback
continue to work. The log counts active layer launches on each card, so the
tradeoff can be measured alongside tokens per second. This placement is
experimental and may be slower if a given layer needs more than one card's
compute capacity.

The experiment requires the corresponding visible CUDA devices and a CUDA
enabled build. Each card must keep at least 512 MiB free. Tiers must be enabled
in order. Under WDDM (Windows, WSL2), CUDA registration of the host expert arena is capped at
8 GiB to leave room for the contexts and MTP on CUDA0 (`STRATA_ARENA_PIN_GIB` overrides it; on Linux the whole
arena is registered). The rest remains
available to the CPU pool; the PCIe expert path is available only for the
registered layers. Without secondary GPUs, the original uncapped registration
behavior applies.

On an existing Linux installation, apply the patch to its source tree, then run:

```sh
./setup.sh --setup --gpu1-experts 5000 --gpu2-experts 5000 --gpu3-experts 5000
```

On Windows, from PowerShell in the installation folder:

```powershell
.\START-HERE.bat --setup --gpu1-experts 5000 --gpu2-experts 5000 --gpu3-experts 5000
```

To test the layer placement instead, add `--gpu-placement layer` to the setup
command. To go back, run the same command with `--gpu-placement stripe`. After
the new engine is built, the mode can also be changed without a rebuild: edit
`--expert-cache-remote-placement` in `strata-iq3_xxs.json` from `layer` to
`stripe` or vice versa and restart the server. If the argument is absent, the
default is `stripe`.

Choose the same model, size, context and image settings as your current
installation when prompted. Setup recompiles the engine, updates its server
configuration, and starts it. Later starts keep those settings. To tune the
slots without recompiling, edit the values after `--expert-cache-device1`,
`--expert-cache-device2` and `--expert-cache-device3` in the stored
`strata-iq3_xxs.json`, then restart with `.\run-iq3_xxs.bat`.

Compare identical requests at one, two and four GPUs, preferably with several
repeats. VRAM use alone does not show useful offload: compare the per-request
CUDA1–3 counts and tokens per second. More GPU contexts and synchronization
may lower the speed. Without a layer split, prompt prefill uses CUDA0 unless
resident-owner prefill is explicitly enabled below. Secondary GPUs otherwise
serve decode, including MTP verification. No peer-to-peer access is required.

Longer contexts reserve more KV/state memory on CUDA0, which reduces its
automatic expert cache. On a 64 GB PC `setup.py` limits IQ3_XXS to 128K even
if 262K was selected; the secondary caches still require the host expert arena.
To try 262K on 64 GB, choose Q2_0 or IQ2_XS instead and check that CUDA0 still
has enough free VRAM after loading. Benchmark long prompts separately from
short decode requests.

The existing warning about the CUDA0 expert-cache GPU hit path still applies:
its outputs diverge from cache-off runs. Treat performance as experimental
until the generated tokens have been validated.

## Optional helper decode optimization

`--remote-expert-opt` (`--serve` only) optimizes the CUDA1-3 helper caches
above. The engine's default is off; since 0.1.39b setup adds it to a config on
two or more GPUs (`--gpus`, or "use both" at start). It acts only when a helper
cache is configured; a layer split runs exactly as before. To leave it out:
setup's `--no-remote-expert-opt`, or `"remote_expert_opt": false` in the
model's `strata-*.json` (kept when setup runs again). Measured by the PR's
author: dual RTX 4090 +63% mixed / +132% code decode over the plain helper
path; RTX 5090 + 4090 +28% / +63%. The primary cache avoids admitting experts already held
by a helper, and helpers replace cold experts with frequently routed CPU
misses using their existing same-layer slots. Each helper reduces its expert
outputs to a weighted partial sum on its GPU before returning one vector per
token. Tokens with no CPU expert work skip CPU activation quantization.

For an existing server configuration with CUDA0 as the primary and CUDA1 as a
helper, use these engine arguments alongside the model and profile arguments:

```text
--expert-cache auto --expert-cache-device1 auto --remote-expert-opt
```

`--expert-cache-device1`, `--expert-cache-device2` and
`--expert-cache-device3` now also accept `auto`: fill each helper from its
assigned ranking using actual aligned expert bytes and free VRAM, retaining
the existing 512 MiB allowance. Explicit numeric budgets still work. This is
startup capacity sizing, not throughput balancing; with several helpers a
card's truncated candidate tail is not redistributed to another card.

The optimization uses the existing host scheduling and pinned-host transport,
not P2P or tensor parallelism. Each layer still waits for its participating
helpers. It changes floating-point summation order, so enabled output is not
claimed to be bitwise identical. Only two-card CUDA operation has been measured
by its author; a community machine measured two cards on HIP (2x RX 6900 XT,
PCIe 4.0 x8 each, IQ3_S): there the expert plan's PCIe share had been taking
experts the helper already held, which made the primary card wait 43 ms per
verify window instead of 19 (decode 40 instead of 66 tok/s). #854 keeps a
helper's experts out of that share; before it, `--pcie-frac 0` avoided the
cost ([bench/results/2026-10-04-rdna2-helper-pcie-share](../bench/results/2026-10-04-rdna2-helper-pcie-share/README.md)).
Three/four cards have not been validated. Without the switch, the
existing decode path remains in use. This does not optimize the separate
`--peer-device` path below or change its existing incompatibility with helper
caches.

## Peer tier (`--peer-device`)

`--peer-device N` puts a second adaptive expert cache on CUDA device N. It
takes the ranked pairs CUDA0's cache does not hold, as many as fit. The peer
computes the rows of its own experts, for decode windows and for prompt
chunks; the activations and the results cross NVLink or another P2P path. The
tier adapts while the server runs, like the primary cache. It is an
alternative to the CUDA1-3 caches above, not a third tier beside them.

- `--peer-device N` (default off): enable the tier on CUDA device N (N >= 1).
- `--peer-reserve-mib M` (default 600): leave M MiB free on the peer card; the
  cache takes what remains. The prompt-path buffers need this headroom.
- `--peer-slots N` (default 0): cap the tier at N experts; 0 = as many as fit.
- `--peer-adapt-swaps N` (default -1): swaps per adaptive round on the peer;
  -1 uses the primary's `--adapt-swaps`.
- `--peer-prefill-rows N` (default -1): the share of each prompt chunk's rows
  the peer computes; -1 is half of chunk x top-k, 0 keeps prompt rows on the
  primary.

Through the server, list both cards in the config's `"gpu"` (e.g. `[0, 1]`, numbered as nvidia-smi numbers them)
and add `--peer-device 1` to its `"args"`; the number is the card's position in that list (with `"gpu": [2, 0]`,
`--peer-device 1` is nvidia-smi's card 0). Several GPUs in `"gpu"` are otherwise a layer split: the server adds
`--layer-split` only when `--peer-device` is not among the args.

`--peer-device` requires `--expert-profile` and an enabled expert cache, and
the device must be visible; it refuses otherwise. It also refuses
`--layer-split` (a different second-GPU mode: use one or the other) and
`--expert-cache-device1..3` (the peer tier already caches experts on that card):

    strata generate: --peer-device cannot be combined with --layer-split (use one or the other)
    strata generate: --peer-device cannot be combined with --expert-cache-device1..3 (the peer tier already caches experts there)

Without `--peer-device` the binary is unchanged; its output is byte-identical
to the release. With `--peer-device` and the same expert set split across the
two cards, the generated tokens are byte-identical to the single-GPU run under
the exactness gate.

With a peer the prompt path keeps the MMQ path; the fused int8 prompt path is
not yet combined with the peer's rows. Mapped host buffers gain
`cudaHostAllocPortable` only with a peer, since only then does a second
context write them. The tier size is manual for now (`--peer-reserve-mib`,
`--peer-slots`); automatic sizing on small cards wants the buffer lending of
#216 and is a follow-up.

## Static two-stage plus decode-helper composition (experimental, HET-032)

The engine admits an explicit two-stage split with a static decode helper on a
visible GPU that runs **no stage**. Helper flags name tier ordinals, not CUDA
ordinals: with the visible order `3090, x8 V100, x4 V100`,
`--split-device 1 --expert-cache-device1 N` puts the later stage on CUDA1 and
the helper on CUDA2; `--split-device 2 --expert-cache-device1 N` reverses those
two V100 roles. CUDA0 runs the early layers. The existing last-stage output-head
and MTP placement is unchanged; the helper owns no attention, recurrent or draft
state.

Both layouts require `--serve`, an explicit `--layer-split K` and
`--split-device`, a static expert profile, numeric helper slots,
`--adapt-swaps 0`, `--expert-worker-contract` and the CPU expert pool. Under
MPS, admission measures and prices each actual device independently: the
stage keeps its existing stage costs/cache budget, while the helper pays its
aligned cache bound, `RemoteExperts::allocation_bytes` buffers and headroom.
An unpriced shared stage/helper device, optimized/weighted remote decode,
automatic helper slots, adaptive peer/refill, pipelined windows, split-skip and
`--prefill-helpers` with a split remain refused.

`--host-native --windowed-experts` retains the required finite host/device
ceilings and `STRATA_PREFILL_RING=8`. Each prompt stage owns a separate source
lane; blocking helper startup fills use lane zero and retain no source pointer.
All decode stages share one helper registry, which drains before the next
dispatch, with two independently accounted ownership groups per stage.
Long prefill executes on the stage GPUs; the decode helper is idle then.
`STRATA_PREFILL_NATIVE_GROUPED=1` may select native arithmetic on each stage
without installing the no-split resident-owner control or prompt-helper buffers.
No-split native helper-on/off ownership bookkeeping is unchanged.

This is an opt-in execution/accounting composition, not a validated profile or
a speed/concurrency claim. Retain neither V100 role assignment without matched
real-model quality/state, memory and decode/prefill measurements; link width
alone is not evidence about its HBM-local arithmetic.

## Resident-owner prefill (experimental, HET-022)

The opt-in native-grouped path has built with inspected SM70/SM86 device code
and exercised both V100 owners on tiny and fresh4K/16K requests. At chunk4096,
matched helper-off/on full first-logit rows are byte-identical at the unchanged
0.01 limit, including held-out tagged retrieval and subsequent fresh recovery;
the exercised GDN/PLE/KV/draft rollback fingerprints also match.
The bounded chunk512 screen answers the objective checks but fails the fixed
cross-chunk first-row limit (maxabs1.2867/1.4495 against chunk4096 on held-out
4K/16K). It is not qualified, and this comparison does not attribute the
difference to helper execution versus inherited chunk arithmetic. Final review
remains; this is not a daily recommendation, an unexercised-context claim or a
128K qualification.

`--prefill-helpers` enables prompt expert work on the static
`--expert-cache-device1..3` owners, with the primary retaining attention, routing
and recurrent state. It requires `--serve`, a positive explicit `--prefill`,
`--expert-cache-remote-placement layer`, and `STRATA_PREFILL_NATIVE_GROUPED=1`.
It does not compose with `--layer-split`, `--peer-device` or optimized remote
decode. The mixed-architecture MMQ helper path failed its numerical comparisons
and is not an available resident-helper default. The qualified experiment uses
fixed residency (`--adapt-swaps 0`) and `STRATA_PREFILL_CPU_SHARE=0`.
For the matched off arm, retain the same native-grouped environment, caches and
decode ownership, omitting only `--prefill-helpers`.

Each layer has at most one remote owner. Overlapping owner layers are rejected,
not silently staged back onto the primary. The owner reads its resident expert
weights, applies the primary's authoritative router weights, and returns
**FP32 weighted partial sums** for the primary's final combine. The native mode
uses the existing native grouped operator on both architectures and reduces
helper rows in router-k order. Its off control computes the same local/owner
partials on the primary and combines them identically. This is an explicitly
different arithmetic configuration from MMQ, not a retroactive pass for failed
MMQ comparisons. It does not stream additional helper weights or use FP16 transport.
Disabling `STRATA_PF_PEER_SUMS` or enabling `STRATA_PF_PEER_STREAM` /
`STRATA_PF_PEER_F16` is rejected. The separate peer tier retains its defaults
and its optional primary-only prompt fallback.

Dedicated remote owners use portable mapped host transport. The existing peer
tier supplies its already-enabled P2P state, including its no-P2P setting:
hardware capability alone never authorizes primary kernels to dereference a
remote cache. The current host's 3090/V100 links have no P2P access.

The owner takes every routed row it holds, bounded by `chunk * 10`. Its FP32
partial-sum return buffer is `chunk * 2560 * 4` bytes, plus activation, routing
and grouping metadata buffers. These pinned-host bytes are charged to the
worker budget, not just reported.

Admission prices every configured remote device's aligned cache bound, base
decode buffers, prompt buffers, MMQ workspace, headroom and explicit opaque
allowances before model allocation. Under MPS it requires explicit cache counts
and independently checks each declared device ceiling. Unpriceable workspaces
refuse. The helper allocator checks its explicit device and pinned-host payloads
against the same counting function; this is not evidence that opaque allocations
or end-to-end peak memory have already been qualified.

Per-prompt helper counters separately report activation, routing/grouping
metadata, result and staged-weight payload bytes, plus device, layers, routed
rows, experts and over-cap work. They count logical payload, not aggregate
bus transactions or cache-line amplification across the two host legs.
The dedicated route must show zero helper-staged weight bytes. Count actual
helper exposure and primary/helper row conservation; do not infer these from
cache size or treat zero helper staging as proof of zero primary uploads.

Both device-specific MMQ tile selection and helper partial-sum reduction can
change arithmetic. Runtime acceptance still needs the existing independent
real-format operator checks, independent weighted-reduction checks,
same-prefix full-model/state and request-isolation checks, and a matched useful
gain. No tolerance is qualified by this implementation or fitted to a candidate's
observed error. The proposed Q8-only parity fixture was removed rather than
presented as coverage of the real native pack.

`native_prefill_reduce_test` checks the native canonical reduction independently
against an exact dyadic CPU oracle at H2560/K10: shuffled partial routing,
assisted/unassisted combination and reused empty helper scratch. This is a
bounded reduction check, not quantized expert-product or full-model acceptance;
the latter still requires the separate matched runtime evidence above.

`STRATA_PREFILL_HELP` / `Prefill::set_stage_helper` is a different mechanism:
it streams non-resident experts into an idle layer-split stage. It remains off
by default and is not evidence for this resident-owner route.
