# #220 final checkpoint — reject on measured performance

The source-derived original-byte IQ2_S planar-codebook candidate was screened
against the actual current decoder on one binary, opt-in off versus on. It is
**rejected on measured primary C1 regression**, not promoted. The coordinator
explicitly accepted this negative disposition with the numerical and routing
limitations below. This record is not full numerical qualification of a kept
candidate.

All owned measurement windows completed and all ten model arms stopped before
the operator hold notification. No GPU or build request was issued afterward.
The operator subsequently took all three GPUs for their own run; do not enqueue,
retry, acquire, launch an engine, change MPS/device state, or rerun this experiment
expecting another outcome. No further GPU command is needed for the rejection.

## Exact next reproducible command — CPU-only readout

```sh
python3 -m json.tool /home/joshkirklin/strata-workspace/scratch/1cat-220/c1-comparison.json
```

The complete study, source map, full-width FFN work/traffic/storage accounting,
predeclared gates, decisions and limitations are in
`docs/benchmarks/1cat-220-iq-decode.json`. `docs/NVIDIA_V100.md` records the bounded
result. The earlier checkpoint remains in git commit `025e45cb` and is also
preserved at `scratch/1cat-220/early-checkpoint.txt`.

## Results

- **Primary C1:** identical32,069-token prefix,192committed outputs per arm,
  observer-free native decode2,095ms off versus2,146ms on: **−2.376514% throughput**.
  Both arms proposed147draft tokens, accepted113 and rejected34; proposal/rejection
  overhead is in committed decode wall. Full192-row logits were not dumped;
  returned response hashes and counts matched.
- Fresh prefill regressions:16K **+1.079939%**,64K **+0.544389%**,32K **+0.560206%**,
  below the declared2% limit. Short8-output decode probes were positive, but were
  not substituted for the predeclared192-output primary.
- **Admitted C2:**192total committed outputs per arm, response-bound monotonic
  window2.409889471s off versus2.416012806s on: **−0.253448% throughput**. Each response
  retained its request ID, usage, own timings, exact returned choices and96outputs.
  Engine logs show slot copying/two-token window capture. Rates do not use the
  one-token admission legs. Both request timings have `draft_n=0`; this is not
  concurrent speculative acceptance/rollback qualification.
- **No held-out:** the primary failed the predeclared useful+3% threshold and
  regressed, so the conditional confirmation was not triggered.
- Every successful arm retained **24,576resident experts**, stopped cleanly and
  stayed within its existing declared limits. Physical host available memory was
  at least41,570,025,472bytes; the8GiB reserve and150/150/420W power limits were
  unchanged. No clock, thermal, fan or MPS-policy changes were made.

## Numerical and exposure scope — do not upgrade it

- **SM70:** independent official-weight ggml dequantization/full-FFN oracle across
  all seven actual-pack pairs passed its established activation-rounding bound.
  Candidate-versus-old GPU complete-expert outputs were bitwise equal; GPU
  dequantization versus official `to_float` differed in0bits. The unquantized FP32
  float FFN is **not** identical to Q8_1 arithmetic: its measured baseline GPU
  relative error was1.07e−2..1.26e−2, not a newly permitted candidate mismatch.
- **SM86:** exact grouped/graph synthetic format and fallback matrix passed.
  The official-pack full-FFN oracle was **not** run there.
- **Full weighted three-GPU model:** all eight emitted predictor rows,1,986,560
  full-vocabulary FP32 logits, were finite and byte-identical, including initial
  and terminal windows. Prefix/token IDs matched. Completed resident diagnostics
  exposed57candidate IQ2 node occurrences at T1/T4; the three SM86 IQ2 occurrences
  retained the old route.
- Hand-built operator groups `[1,3,4,5,0,8,2,11,1]` supplied35actual entries.
  Empty/uneven/tail groups, duplicate token routes, unused output maps, changed
  graph inputs/counts and interleaved request-private workspaces passed exact
  control comparison on SM70 off/on and SM86.
- Direct intermediate-payload/padding equality was **not separately dumped**.
  There was no independent weighted-combine operator oracle; complete weighted
  model logits provide the recorded end-to-end evidence. Sampler, MTP state and
  rollback code were not changed.
- **No natural-M histogram retained:** all four default/attempted census files
  are empty. The default zero-doorbell path bypasses the host dump hook. The
  attempted override used `STRATA_VERIFY_RESIDENT=0` from a stale adjacent comment;
  pinned `src/core/verify.cpp:1391` actually reads `STRATA_VERIFY_ALL_RESIDENT`.
  That attempted override was inert and did not force host planning. Correct
  census is **unexercised**, not proved unobtainable. Empty evidence is preserved.
  T1/T4 is verification width, not GEMM M; the hand-built operator set is not a
  sampled workload distribution. No cross-planning exactness is claimed.

The rejection does not depend on the missing SM86/payload/census evidence, but
this record must never be cited as having supplied that evidence. One matched
pair has no variance estimate or general slowdown guarantee.

## Unexercised correct census command — future authorization only

Source-prepared `profile-census-correct-0.json` uses the actual
`STRATA_VERIFY_ALL_RESIDENT=0` control and fresh output paths. It has not been
executed or qualified. Do **not** run this while the operator owns the GPUs.
It is a future capability probe, not needed to reject the measured regressing
candidate, and admission/numerical success is not assumed:

```sh
/home/joshkirklin/ai-optimise/tools/host-inference/window.py run --class measure --ticket 220 --purpose "Unexercised correct natural-M census, only after operator authorization" --max-s 1800 -- /home/joshkirklin/ai-optimise/.venv/bin/python /home/joshkirklin/ai-optimise/tools/host-inference/measure.py --profile /home/joshkirklin/strata-workspace/scratch/1cat-220/profile-census-correct-0.json --run-id iq220-census-correct-off-r1 --out /home/joshkirklin/strata-workspace/scratch/1cat-220/census-correct-off-r1.json --request-json /home/joshkirklin/strata-workspace/scratch/1cat-220/fixture-1024.json
```


## Source and binary identities

- Branch `1cat/220-iq-decode`; base
  `32e14d4febdd026b5f50901c8c76b5be0b128ac5`.
- Implementation/study source
  `c7bccc9a48a70322ca6e7a1625c70dafcff9350b`.
- Engine `/home/joshkirklin/strata-workspace/scratch/1cat-220-build/strata`, SHA256
  `1bedc578776d2fb8523c7998e8454848697bb7b7ea06e617a6cd155f42401bed`.
- Official-weight oracle SHA256
  `91f3c235e5db3cde1389307c232f8058daf2627d042af80672e097aad873457a`.
- Grouped/graph fixture SHA256
  `d81b78eee90c86abaad01f6b0c268576d4b049ab36621006232b6a953a63d94c`.
- Successful queued build `d818e45b6a6d`, exit0,309.56s: CUDA12.9.86, conda GNU14.4.0,
  Release, SM70+SM86, native experts/tests and experimental SM60 enabled; external
  ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Make generator.
- Original adopted control remains untouched:
  `/home/joshkirklin/strata-workspace/scratch/s043/strata-repin-32e14d4f`, SHA256
  `5862438577303ac56ce4f385499100732f0e65ccd42540021991c3d4a7868b68`.
- Rollback: leave `STRATA_IQ2_PLANAR_BOOK` unset/0. No weight/profile conversion,
  permanent expert bank, sampler change or daily-profile change occurred. Candidate
  is retained only as an unpromoted experiment for evidence reproduction.

## Exact historical commands and retained artifacts

Scratch root: `/home/joshkirklin/strata-workspace/scratch/1cat-220`.

- `build-output.txt`, build directory `CMakeCache.txt`/configure logs: successful
  build and preparation failures. The full configure pinned C/CXX/CUDA host
  compilers, compiler AR/RANLIB, `CUDAToolkit_ROOT`, architecture70;86, Release,
  CUDA/native experts/tests/experimental SM60 ON and the external ggml checkout.
  Built targets: `strata native_expert_parity native_expert_bench native_grouped_parity iq_multi_parity`.
- `oracle-sm70-on.json`, `off-sm86-parity.json`: structured transcriptions of
  queue `c1ee309021f2`/`b7557e552d5a`; primary output is retained in agent history.
  `diag-window.log` includes the missing seventh official pair (layer20).
- `diag-window.log`, `c1-window.log`, `route-cohort-window.log`: each records exact
  complete `window.py run` argv, purpose, every transient refusal, grant, real exit
  and stdout. These are historical commands, **not requests to run them now**.
- Frozen `fixture-1024.json`:1024prompt tokens; request SHA256
  `db8c3fda9a49a8a29923e86acc54b46d5d199f663616c944824b0564ccee039b`.
- `profile-0.json`/`profile-1.json`: same binary, only opt-in differs. Versus daily:
  binary path,900s lifetime, prompt-cache0, no-prefill-borrow (placement reserve),
  scratch artifact root and opt-in. Hardware/model/frontend/batch configuration
  remains identical between the latency arms.
- `profile-diag-*.json`: separately labelled serial diagnostics remove batching
  flags for the supported complete-committed-logit surface. `profile-route-*.json`
  attempted census with an inert, incorrectly named override; their empty files
  and duplicate resident-path exact logit repeats remain.
- `cheap-comparison.json`, `c1-comparison.json`, `cohort-comparison.json`,
  `complete-logits-comparison.json`, `route-capability-finding.json`,
  `lifecycle-safety.json`: completed CPU readouts linked to raw results.
- Raw model runs: `scratch/1cat-220-runs/iq220-*`; all logs, config/telemetry,
  statuses, complete-logit binaries and failed/empty routing evidence retained.
- `cohort.py`: two-request response-bound adapter using the existing
  `measure.py.run_arm` launch/telemetry/stop; `queue_run.py`: authorized transient
  queue backoff only, never an executed-operation retry. No wrapper is left running.

Nothing was pushed, promoted or deleted. The earlier transient-refusal checkpoint
was superseded by completed measurements; operator machine ownership is respected.
